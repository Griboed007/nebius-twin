#!/usr/bin/env bash
# Deploy the twin API as a Nebius Serverless Endpoint (CPU). DRY-RUN by default —
# prints the command (token REDACTED). Set DEPLOY_CONFIRM=1 to deploy.
# WARNING: an Endpoint is PUBLIC and bills continuously (no autoscaling yet).
# Spin it up only for the demo and tear it down right after: bash deploy/teardown.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; }
: "${NEBIUS_PROJECT_ID:?}"; : "${NEBIUS_SUBNET_ID:?}"; : "${ENDPOINT_AUTH_TOKEN:?}"
IMAGE="${1:?usage: endpoint.sh <image> (e.g. cr.eu-north1.nebius.cloud/<reg>/twin-api:latest)}"

# ALWAYS pin CPU — create defaults to gpu-h100-sxm.
PLATFORM="${NEBIUS_PLATFORM:-cpu-e2}"; PRESET="${NEBIUS_PRESET:-2vcpu-8gb}"
# Where 040 wrote the twin in the bucket (mounted at /data). Local default for no-bucket runs.
TWIN_TTL_PATH="${TWIN_TTL_PATH:-/data/out/twin.ttl}"

cmd=(nebius ai endpoint create
  --name twin-api
  --parent-id "$NEBIUS_PROJECT_ID"
  --image "$IMAGE"
  --container-port 8080
  --platform "$PLATFORM" --preset "$PRESET"
  --auth token --token "$ENDPOINT_AUTH_TOKEN"
  --public
  --subnet-id "$NEBIUS_SUBNET_ID"
  --env "ENDPOINT_AUTH_TOKEN=${ENDPOINT_AUTH_TOKEN}"
  --env "TWIN_TTL_PATH=${TWIN_TTL_PATH}"
  --env "MODEL_PATH=${MODEL_PATH:-}")
[ -n "${NEBIUS_BUCKET:-}" ] && cmd+=(--volume "${NEBIUS_BUCKET}:/data:rw")

# Echo with the token redacted so it never lands in a terminal/transcript/log.
disp=("${cmd[@]}")
for i in "${!disp[@]}"; do disp[$i]="${disp[$i]//$ENDPOINT_AUTH_TOKEN/***REDACTED***}"; done
printf '+'; printf ' %q' "${disp[@]}"; echo

if [ "${DEPLOY_CONFIRM:-0}" = "1" ]; then
  "${cmd[@]}"
  echo "# Deployed (PUBLIC, billing). Capture proof, then TEAR DOWN: bash deploy/teardown.sh"
  echo "#   URL=http://<public-ip>:8080   (from the create output above)"
  echo "#   curl -s \$URL/health"
  echo "#   curl -s -H 'Authorization: Bearer <token>' -H 'Content-Type: application/json' \\"
  echo "#     -d '{\"query\":\"SELECT (COUNT(?o) AS ?n) WHERE {?o a <https://saref.etsi.org/core/Observation>}\"}' \$URL/twin/sparql"
  echo "#   (save outputs to docs/proof/050-endpoint-live.txt — REDACT the token)"
else
  echo "DRY-RUN — set DEPLOY_CONFIRM=1 to deploy a PUBLIC endpoint (spends \$). Nothing deployed."
fi
