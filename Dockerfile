# ─────────────────────────────────────────────────────────────────────────────
# Credit Risk Intelligence Platform — Dockerfile
# Agent 3 | Vaidik Sharma | github.com/Vaidik6920
#
# Multi-stage build:
#   Stage 1 (builder): install dependencies into /install
#   Stage 2 (runtime): slim image, non-root user, only runtime deps
#
# Build:  docker build -t credit-risk-api:latest .
# Run:    docker run -p 8000:8000 credit-risk-api:latest
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /install

# System build deps (needed for LightGBM, XGBoost, CatBoost wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (Docker layer cache)
COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -m appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install/lib /usr/local/lib
COPY --from=builder /install/bin /usr/local/bin

# Copy application code
COPY api/        ./api/
COPY src/        ./src/
COPY configs/    ./configs/
COPY models/     ./models/

# Create data directories (runtime artifacts)
RUN mkdir -p data/plots data/processed \
    && chown -R appuser:appuser /app

# Switch to non-root
USER appuser

# Environment
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

EXPOSE 8000

# Health check — hits /health every 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Entrypoint — Gunicorn + Uvicorn workers for production
CMD ["sh", "-c", \
     "uvicorn api.main:app \
      --host 0.0.0.0 \
      --port ${PORT} \
      --workers 2 \
      --timeout-keep-alive 30 \
      --log-level info"]
