# Lean image for the twin_api endpoint (Nebius Serverless container, port 8080).
# Base: slim Python to keep cold starts fast; no GPU needed (CPU inference only).
FROM python:3.11-slim

# Suppress pip version warnings for leaner build output
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python dependencies first (layer-cached on requirements change)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source — contracts, src/, services/ (no test/ or deploy/ in image)
COPY contracts/ ./contracts/
COPY src/ ./src/
COPY services/twin_api/ ./services/twin_api/

# Runtime: single worker, port 8080 (Nebius Serverless Endpoint default)
EXPOSE 8080
CMD ["uvicorn", "services.twin_api.app:app", "--host", "0.0.0.0", "--port", "8080"]
