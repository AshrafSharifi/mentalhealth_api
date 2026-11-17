# ---- Base image: small Debian with Python 3.12 (same version as local dev) ----
FROM python:3.12-slim

# No .pyc files, unbuffered logs (so print/log lines appear immediately), no pip cache
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_PATH=/app/models/model.joblib

WORKDIR /app

# ---- Dependencies first: this layer is cached until requirements.txt changes ----
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---- Application code and the trained model ----
COPY api/ ./api/
COPY models/model.joblib ./models/model.joblib

# ---- Run as a non-root user ----
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

# Docker marks the container "unhealthy" if /health stops answering.
# (curl is not installed in slim images, so we use Python's standard library.)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]