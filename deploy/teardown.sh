#!/usr/bin/env bash
# Delete the twin Job + Endpoint to protect the $100 credit. Idempotent, list-based.
# `nebius ai <kind> delete` takes --id (NOT --name), so we resolve names -> ids.
# Usage: teardown.sh            # deletes twin-api + twin-batch-enrich
#        teardown.sh ALL        # deletes EVERY endpoint+job in the project
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; }
: "${NEBIUS_PROJECT_ID:?set NEBIUS_PROJECT_ID in .env}"
TARGETS="${1:-twin-api twin-batch-enrich}"

json="$(nebius ai list --parent-id "$NEBIUS_PROJECT_ID" --format json 2>/dev/null || true)"

mapfile -t PAIRS < <(printf '%s' "$json" | python3 - "$TARGETS" <<'PY'
import sys, json, re
targets = sys.argv[1].split()
raw = sys.stdin.read()
try:
    data = json.loads(raw)
except Exception:
    # Fallback: pull any ids out of whatever text the CLI returned.
    for rid in dict.fromkeys(re.findall(r'ai(?:endpoint|job)-[a-z0-9]+', raw)):
        print(f"{rid}\t?")
    sys.exit(0)
found = []
def walk(o):
    if isinstance(o, dict):
        rid = o.get('id') or (o.get('metadata') or {}).get('id')
        name = o.get('name') or (o.get('metadata') or {}).get('name')
        if isinstance(rid, str) and re.fullmatch(r'ai(endpoint|job)-[a-z0-9]+', rid):
            found.append((rid, name or '?'))
        for v in o.values():
            walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)
walk(data)
seen = set()
for rid, name in found:
    if rid in seen:
        continue
    seen.add(rid)
    if targets == ['ALL'] or name in targets:
        print(f"{rid}\t{name}")
PY
)

if [ "${#PAIRS[@]}" -eq 0 ]; then
  echo "nothing to tear down (no matching endpoints/jobs in $NEBIUS_PROJECT_ID)."
  exit 0
fi
for line in "${PAIRS[@]}"; do
  id="${line%%$'\t'*}"; name="${line#*$'\t'}"
  kind="job"; [ "${id#aiendpoint-}" != "$id" ] && kind="endpoint"
  echo ">> deleting $kind $name ($id)"
  nebius ai "$kind" delete --id "$id" || true
done
echo "torn down. Verify: nebius ai list --parent-id $NEBIUS_PROJECT_ID  (and the console)."
