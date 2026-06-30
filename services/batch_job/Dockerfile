# Lean batch-job image (proposal 040).
# CPU-only; torch is not installed (residual falls back to no-op cleanly).
# Cold-start target: keep layers minimal.
#
# Env vars at runtime:
#   INPUT_URI    — source trajectory parquet (local path or mounted bucket)
#   OUTPUT_URI   — destination directory for enriched.parquet / twin.ttl / metrics.json
#   MODEL_PATH   — optional residual .pt file; omit for baseline-only mode

FROM python:3.11-slim

# System deps: only what pyarrow/rdflib need at runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (no torch — keep image small; fastapi/uvicorn not needed here)
RUN pip install --no-cache-dir \
        "pydantic>=2.7" \
        "numpy>=1.26" \
        "pandas>=2.2" \
        "pyarrow>=16.0" \
        "rdflib>=7.0" \
        "scikit-learn>=1.4" \
    && rm -rf /root/.cache/pip

# Copy source
COPY contracts/ contracts/
COPY src/ src/
COPY services/batch_job/ services/batch_job/

# Env defaults (overridden at runtime via -e or Nebius Job env)
ENV INPUT_URI=./artifacts/synthetic.parquet \
    OUTPUT_URI=./artifacts/out \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["python3", "-m", "services.batch_job.run"]
