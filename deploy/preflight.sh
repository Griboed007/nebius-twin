#!/usr/bin/env bash
# One-time preflight for the LIVE Nebius deploy. Read-only: checks env + bucket
# setup and prints the exact remediation commands. Spends nothing.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; }

ok=1
echo "== required env =="
for v in NEBIUS_PROJECT_ID NEBIUS_SUBNET_ID NEBIUS_REGISTRY ENDPOINT_AUTH_TOKEN INPUT_URI OUTPUT_URI; do
  if [ -z "${!v:-}" ]; then echo "  MISSING  $v"; ok=0; else echo "  ok       $v"; fi
done
# NEBIUS_BUCKET is only required if you mount a bucket (live run with /data paths).
if [ -z "${NEBIUS_BUCKET:-}" ]; then
  echo "  note     NEBIUS_BUCKET unset — fine for local paths; REQUIRED for a real Job (set s3://<bucket>)"
fi

echo
echo "== bucket S3 creds (~/.aws) for the --volume mount =="
if [ -f "$HOME/.aws/credentials" ] && [ -f "$HOME/.aws/config" ]; then
  echo "  ok       ~/.aws/credentials and ~/.aws/config present"
else
  ok=0
  echo "  MISSING  ~/.aws/{credentials,config}. Create an access key + config:"
  echo "             nebius iam v2 access-key create --name dt-s3 --service-account-id <sa-id>"
  echo "             # write the key id/secret to ~/.aws/credentials [default]"
  echo "             # ~/.aws/config:  region=eu-north1"
  echo "             #                 endpoint_url=https://storage.eu-north1.nebius.cloud"
fi

echo
echo "== seed the input into the bucket (image does not bake artifacts/) =="
echo "   make synth"
echo "   aws s3 cp artifacts/synthetic.parquet \"${NEBIUS_BUCKET:-s3://<bucket>}/synthetic.parquet\" \\"
echo "       --endpoint-url https://storage.eu-north1.nebius.cloud"
echo "   # then in .env: INPUT_URI=/data/synthetic.parquet  OUTPUT_URI=/data/out"

echo
if [ "$ok" = 1 ]; then
  echo "PREFLIGHT OK — build/push then deploy:"
  echo "  bash deploy/build_push.sh"
  echo "  DEPLOY_CONFIRM=1 bash deploy/job.sh      <job-image>"
  echo "  DEPLOY_CONFIRM=1 bash deploy/endpoint.sh <api-image>"
  echo "  bash deploy/teardown.sh   # after capturing proof"
else
  echo "PREFLIGHT INCOMPLETE — resolve the MISSING items above first."
  exit 1
fi
