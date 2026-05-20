FROM python:3.11-slim

# Disable .pyc files; keep stdout/stderr unbuffered for live log streaming
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOG_DIR=/app/logs

WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
# libgomp1       – OpenMP runtime required by numpy / spaCy vectorised ops
# ca-certificates – TLS trust store for azure-identity and Purview HTTPS calls
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies (separate layer for cache efficiency) ─────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── spaCy model (downloaded at build time; no network needed at runtime) ──────
RUN python -m spacy download en_core_web_lg \
    && pip cache purge

# ── Application code ──────────────────────────────────────────────────────────
COPY main.py .

# ── Non-root user (uid/gid 1001) ──────────────────────────────────────────────
# Created AFTER pip/spacy steps so package installation runs as root and the
# installed wheels remain readable by all users (755 dirs, 644 files).
# /app/logs is owned by appuser so the process can write audit files there.
RUN groupadd --gid 1001 appgroup \
    && useradd  \
        --uid 1001 \
        --gid 1001 \
        --no-create-home \
        --no-log-init \
        --shell /bin/sh \
        appuser \
    && mkdir -p /app/logs \
    && chown -R appuser:appgroup /app/logs

# Drop privileges — all subsequent layers and the final CMD run as appuser
USER appuser

# Mount a volume here to persist audit logs across container restarts:
#   docker run -v /host/logs:/app/logs ...
VOLUME ["/app/logs"]

CMD ["python", "main.py"]
