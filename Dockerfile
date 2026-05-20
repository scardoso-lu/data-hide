# ---------------------------------------------------------------------------
# Stage: runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Keeps Python from writing .pyc files and buffers stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# libgomp1  – OpenMP runtime required by numpy / spaCy vectorised ops
# ca-certificates – needed by azure-identity HTTPS calls inside the container
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer is cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the spaCy NLP model required by Presidio.
# en_core_web_lg provides high-accuracy NER; the download is done at build
# time so the image is fully self-contained and requires no network at runtime.
RUN python -m spacy download en_core_web_lg \
    && pip cache purge

# Copy application code
COPY main.py .

# Run as a non-root user for least-privilege security
RUN useradd --create-home --no-log-init --shell /bin/false appuser \
    && chown -R appuser:appuser /app
USER appuser

CMD ["python", "main.py"]
