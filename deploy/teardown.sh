#!/usr/bin/env bash
# Delete the twin Job + Endpoint to protect the $100 credit. Idempotent.
# `nebius ai <kind> delete` takes --id (NOT --name), so we resolve names -> ids
# from the plain `nebius ai list` table: col1=Type(endpoint|job) col2=ID col3=Name.
# (The --format json shape proved unreliable to parse; the table is stable.)
# Usage: teardown.sh            # deletes twin-api + twin-batch-enrich
#        teardown.sh ALL        # deletes EVERY endpoint+job in the project
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; }
: "${NEBIUS_PROJECT_ID:?set NEBIUS_PROJECT_ID in .env}"
TARGETS="${1:-twin-api twin-batch-enrich}"

# Rows of: "<kind> <id> <name>" for matching resources (kind == delete subcommand).
mapfile -t ROWS < <(
  nebius ai list --parent-id "$NEBIUS_PROJECT_ID" 2>/dev/null \
    | awk -v targets="$TARGETS" '
        BEGIN { all = (targets == "ALL"); n = split(targets, t, " "); for (i=1;i<=n;i++) want[t[i]]=1 }
        $2 ~ /^ai(endpoint|job)-/ { if (all || ($3 in want)) print $1, $2, $3 }
      '
)

if [ "${#ROWS[@]}" -eq 0 ]; then
  echo "nothing to tear down (no matching endpoints/jobs in $NEBIUS_PROJECT_ID)."
  exit 0
fi
for row in "${ROWS[@]}"; do
  kind="${row%% *}"; rest="${row#* }"; id="${rest%% *}"; name="${rest#* }"
  echo ">> deleting $kind $name ($id)"
  # Filter any auth_token the CLI may echo for endpoints; never surface secrets.
  nebius ai "$kind" delete --id "$id" 2>&1 | grep -viE 'auth_token|token:|password' || true
done
echo "torn down. Verify: nebius ai list --parent-id $NEBIUS_PROJECT_ID  (and the console)."
