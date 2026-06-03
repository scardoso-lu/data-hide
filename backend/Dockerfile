FROM python:3.11-slim

# Disable .pyc bytecode; keep stdout/stderr unbuffered for live log streaming.
# UV_LINK_MODE=copy avoids hardlink warnings when the cache and target dirs
# live on different filesystems (common in Docker layered builds).
#
# MALLOC_TRIM_THRESHOLD_=131072 makes glibc return freed heap pages to the OS
# at a 128 KB threshold instead of its adaptive default (which can grow into
# the tens of MB and keep large freed blocks mapped).  Combined with the
# explicit malloc_trim(0) calls after each model release, this keeps RSS
# tracking actual live usage so the container frees memory between the
# language-classification passes and between tables, rather than plateauing
# at the high-water mark.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SPACY_MODELS_DIR=/app/models \
    MALLOC_TRIM_THRESHOLD_=131072 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
# libgomp1        – OpenMP runtime required by numpy / spaCy vectorised ops
# ca-certificates – TLS trust store for azure-identity, Purview, and webhook calls
# msodbcsql18     – Microsoft ODBC Driver 18 for SQL Server (Fabric SQL endpoint)
# unixodbc-dev    – headers needed by pyodbc at install time
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
        curl \
        gnupg \
    && curl -sSL https://packages.microsoft.com/keys/microsoft.asc \
        | gpg --dearmor -o /usr/share/keyrings/microsoft.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft.gpg] \
https://packages.microsoft.com/debian/12/prod bookworm main" \
        > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y --no-install-recommends \
        msodbcsql18 \
        unixodbc-dev \
    && rm -rf /var/lib/apt/lists/*

# ── uv binary ────────────────────────────────────────────────────────────────
# Pulled from the official Astral image, verified by digest in the published
# manifest.  Pinned to a major version to limit supply-chain exposure;
# bump deliberately and re-run `uv lock` when upgrading.
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /usr/local/bin/

# ── Python dependencies (own layer — cached unless lock files change) ─────────
# --frozen     refuses to update uv.lock; fails the build if the lock is stale,
#              guaranteeing the image matches what was committed and reviewed.
# --no-dev     skips pytest etc., keeping the runtime image lean.
# --no-install-project is defensive: pyproject sets `package = false` already.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# ── Application code ──────────────────────────────────────────────────────────
COPY app ./app

# ── Non-root user (uid/gid 1001) ──────────────────────────────────────────────
# /app/models is writable by appuser so spaCy models can be downloaded at
# first startup and reused across runs via a mounted volume.
RUN groupadd --gid 1001 appgroup \
    && useradd \
        --uid 1001 \
        --gid 1001 \
        --no-create-home \
        --no-log-init \
        --shell /bin/sh \
        appuser \
    && mkdir -p /app/models \
    && chown appuser:appgroup /app/models

USER appuser

# Health probe endpoints (/healthz/live, /healthz/startup, /healthz/ready)
# served by app/health.py — see deploy/containerapp.yaml for the Azure
# Container Apps probe configuration. Override the port with HEALTH_PORT.
EXPOSE 8080

CMD ["python", "-m", "app.main"]
