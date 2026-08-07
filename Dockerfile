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

# Create data directories
RUN mkdir -p /app/server/uploads /app/server/outputs /app/server/tm_thumbs

ENV PYTHONUTF8=1
ENV TOFU_FRONTEND_URL=http://localhost:3000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/capabilities')" || exit 1

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app/server"]
