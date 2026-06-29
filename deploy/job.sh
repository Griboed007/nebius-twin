#!/usr/bin/env bash
# Run the batch-enrichment container as a Nebius Serverless Job. Env from .env.
set -euo pipefail
: "${NEBIUS_PROJECT_ID:?}"; : "${INPUT_URI:?}"; : "${OUTPUT_URI:?}"
IMAGE="${1:?usage: job.sh <image_path>}"
nebius ai job create \
  --name twin-batch-enrich \
  --image "$IMAGE" \
  --platform cpu-e2 \
  --env "INPUT_URI=${INPUT_URI}" \
  --env "OUTPUT_URI=${OUTPUT_URI}" \
  --env "MODEL_PATH=${MODEL_PATH:-}" \
  --subnet-id "${NEBIUS_SUBNET_ID:-}"
# Capture logs to docs/proof/ after completion (no secrets).
