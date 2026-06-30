#!/usr/bin/env bash
# Build the 040 job + 050 endpoint images and push them to the Nebius registry.
# Prepared, not destructive: this only builds/pushes images (no compute spend).
#
# Env (from .env): NEBIUS_REGISTRY = registry id WITHOUT the 'registry-' prefix
#   (e.g. e00y0bsxm2b24wq5ra). Full path = ${NEBIUS_CR_HOST}/${NEBIUS_REGISTRY}.
# Repo root is the docker build context (Dockerfiles COPY contracts/ src/ services/).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; }
: "${NEBIUS_REGISTRY:?set NEBIUS_REGISTRY (registry id) in .env}"

CR_HOST="${NEBIUS_CR_HOST:-cr.eu-north1.nebius.cloud}"
TAG="${IMAGE_TAG:-latest}"
BASE="${CR_HOST}/${NEBIUS_REGISTRY}"

# One-time docker auth for the Nebius CR (idempotent; safe to re-run).
# If this errors, run it manually: `nebius registry configure-helper ${CR_HOST}`.
nebius registry configure-helper "$CR_HOST" >/dev/null 2>&1 || \
  echo "WARN: could not auto-configure docker cred helper for ${CR_HOST}; run 'nebius registry configure-helper ${CR_HOST}' if push is denied."

build_push () { # <dockerfile> <name>
  local df="$1" name="$2" img="${BASE}/${name}:${TAG}"
  echo ">> build $img"
  docker build -t "$img" -f "$df" "$ROOT"
  echo ">> push  $img"
  docker push "$img"
  echo "$img"
}

JOB_IMG=$(build_push "$ROOT/services/batch_job/Dockerfile" twin-job)
API_IMG=$(build_push "$ROOT/services/twin_api/Dockerfile"  twin-api)

echo
echo "JOB_IMAGE=$JOB_IMG"
echo "API_IMAGE=$API_IMG"
echo "# Pass these to: bash deploy/job.sh \$JOB_IMAGE   and   bash deploy/endpoint.sh \$API_IMAGE"
