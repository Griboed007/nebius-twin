#!/usr/bin/env bash
# Run the batch-enrichment container as a Nebius Serverless Job (CPU, cheap).
# DRY-RUN by default — prints the exact command. Set DEPLOY_CONFIRM=1 to create
# (this spends the $100 credit). A Job is finite: it completes and stops on its own.
#
# PREFLIGHT (one-time — see deploy/preflight.sh and docs/proof/part-a-job.txt):
#   1. Bucket S3 creds in ~/.aws/{credentials,config} for the --volume mount
#      (nebius iam v2 access-key create ; region eu-north1 ;
#       endpoint_url https://storage.eu-north1.nebius.cloud).
#   2. Seed the input INTO the bucket — the image does NOT bake artifacts/ and
#      INPUT_URI points at the mounted path:
#        aws s3 cp artifacts/synthetic.parquet "$NEBIUS_BUCKET/" --endpoint-url https://storage.eu-north1.nebius.cloud
#   For a live run set in .env: NEBIUS_BUCKET=s3://<bucket>, INPUT_URI=/data/synthetic.parquet, OUTPUT_URI=/data/out
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; }
: "${NEBIUS_PROJECT_ID:?}"; : "${NEBIUS_SUBNET_ID:?}"; : "${INPUT_URI:?}"; : "${OUTPUT_URI:?}"
IMAGE="${1:?usage: job.sh <image> (e.g. cr.eu-north1.nebius.cloud/<reg>/twin-job:latest)}"

# ALWAYS pin CPU — `nebius ai job create` defaults to gpu-h100-sxm, which burns credit fast.
PLATFORM="${NEBIUS_PLATFORM:-cpu-e2}"; PRESET="${NEBIUS_PRESET:-2vcpu-8gb}"

cmd=(nebius ai job create
  --name twin-batch-enrich
  --parent-id "$NEBIUS_PROJECT_ID"
  --image "$IMAGE"
  --platform "$PLATFORM" --preset "$PRESET"
  --subnet-id "$NEBIUS_SUBNET_ID"
  --env "INPUT_URI=${INPUT_URI}"
  --env "OUTPUT_URI=${OUTPUT_URI}"
  --env "MODEL_PATH=${MODEL_PATH:-}")
# Bucket mount (required when INPUT_URI/OUTPUT_URI are /data/... paths):
[ -n "${NEBIUS_BUCKET:-}" ] && cmd+=(--volume "${NEBIUS_BUCKET}:/data:rw")
# Private-CR pull is automatic same-tenant; uncomment only if pull is denied:
#   cmd+=(--registry-username iam --registry-password "$(nebius iam get-access-token)")

printf '+'; printf ' %q' "${cmd[@]}"; echo
if [ "${DEPLOY_CONFIRM:-0}" = "1" ]; then
  "${cmd[@]}"
  echo "# Submitted. Capture proof (no secrets) once COMPLETED:"
  echo "#   nebius ai list --parent-id $NEBIUS_PROJECT_ID"
  echo "#   nebius ai job logs <job-id> --parent-id $NEBIUS_PROJECT_ID > docs/proof/040-job-live.txt"
else
  echo "DRY-RUN — set DEPLOY_CONFIRM=1 to create (spends \$ credit). Nothing submitted."
fi
