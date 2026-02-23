# ──────────────────────────────────────────────────────────────
# Banking & Insurance ML Platform — Dockerfile
# HuggingFace Spaces compatible (port 7860, non-root user)
# ──────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create non-root user (required by HuggingFace Spaces)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

CMD ["gunicorn", "app:create_app()", \
     "--bind", "0.0.0.0:7860", \
     "--workers", "2", \
     "--threads", "4", \
     "--timeout", "120", \
     "--worker-class", "sync"]
