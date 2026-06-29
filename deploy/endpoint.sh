#!/usr/bin/env bash
# Deploy the twin API container as a Nebius Serverless Endpoint. Env from .env.
set -euo pipefail
: "${ENDPOINT_AUTH_TOKEN:?}"
IMAGE="${1:?usage: endpoint.sh <image_path>}"
nebius ai endpoint create \
  --name twin-api \
  --image "$IMAGE" \
  --container-port 8080 \
  --platform cpu-e2 \
  --auth token --token "$ENDPOINT_AUTH_TOKEN" \
  --public \
  --subnet-id "${NEBIUS_SUBNET_ID:-}"
# Record the endpoint URL + a SPARQL response to docs/proof/, then stop it.
