FROM python:3.11-slim

# Disable .pyc bytecode; keep stdout/stderr unbuffered for live log streaming
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
# libgomp1        – OpenMP runtime required by numpy / spaCy vectorised ops
# ca-certificates – TLS trust store for azure-identity, Purview, and webhook calls
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies (own layer — cached unless requirements.txt changes) ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── spaCy model (baked in at build time; no network needed at runtime) ─────────
RUN python -m spacy download en_core_web_lg \
    && pip cache purge

# ── Application code ──────────────────────────────────────────────────────────
COPY app ./app

# ── Non-root user (uid/gid 1001) ──────────────────────────────────────────────
# Created after pip/spaCy steps so package installation still runs as root and
# the installed files remain world-readable (755 dirs / 644 files).
# The container writes nothing to disk at runtime — no volume is needed.
RUN groupadd --gid 1001 appgroup \
    && useradd \
        --uid 1001 \
        --gid 1001 \
        --no-create-home \
        --no-log-init \
        --shell /bin/sh \
        appuser

USER appuser

CMD ["python", "-m", "app.main"]
