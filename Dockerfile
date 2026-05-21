FROM python:3.11-slim

# Disable .pyc bytecode; keep stdout/stderr unbuffered for live log streaming
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SPACY_MODELS_DIR=/app/models

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

# ── Python dependencies (own layer — cached unless requirements.txt changes) ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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

CMD ["python", "-m", "app.main"]
