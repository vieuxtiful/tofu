# ─── frontend build stage ─────────────────────────────────────────────────
FROM node:20-slim AS frontend-build
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npx tsc --noEmit && npm run build

# ─── app stage ────────────────────────────────────────────────────────────
FROM python:3.13-slim AS app

# System dependencies: ffmpeg for video, libglib for headless OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg ffprobe libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install the library itself
COPY pyproject.toml README.md /app/
COPY src/ /app/src/
RUN pip install --no-cache-dir -e ".[all]"

# Copy server code
COPY server/ /app/server/

# Copy scripts (paddle_worker, inpaint_lama_runner, eval harnesses)
COPY scripts/ /app/scripts/

# Copy frontend build output
COPY --from=frontend-build /build/dist /app/frontend/dist

# Pull the recognizer's weights now rather than on the first user request.
# Kept after the source copies so editing code does not re-download 250 MB,
# and before the data directories so a warm failure fails the build loudly.
ENV PYTHONUTF8=1
RUN python /app/scripts/warm_easyocr.py

# Create data directories
RUN mkdir -p /app/server/uploads /app/server/outputs /app/server/tm_thumbs

## Where the side-loaded language artifacts are expected. The directories are
## created empty and are a bind-mount point in docker-compose.prod.yml: the
## n-gram and vector models are built once by the operator (see
## docs/deployment.md) rather than baked, because they are derived from a
## licensed corpus whose provenance belongs in the deployment record, not
## anonymously inside a layer.
##
## Empty is a supported state. Every provider in tofu.layers.language_models
## reports not-ready and returns None, arbitration renormalizes the missing
## weight away, and the deployment behaves exactly as it did before these
## variables existed.
ENV TOFU_NGRAM_DIR=/app/models/ngram
ENV TOFU_VECTOR_DIR=/app/models/vectors
RUN mkdir -p /app/models/ngram /app/models/vectors

ENV TOFU_FRONTEND_URL=http://localhost:3000
EXPOSE 8000

## /api/health, not /api/capabilities. The capability inventory enumerates
## installed models and filesystem paths, so security.py keeps it behind the
## API key -- a healthcheck pointed at it reports every production container
## as unhealthy the moment TOFU_API_KEYS is set, which is exactly when the
## deployment is most load-bearing. /api/health is the unauthenticated
## posture route that exists for this.
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app/server"]
