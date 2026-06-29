#!/usr/bin/env bash
# Delete Nebius resources to protect the $100 credit. Idempotent.
set -euo pipefail
nebius ai endpoint delete --name twin-api || true
nebius ai job delete --name twin-batch-enrich || true
echo "torn down (verify in console)."
