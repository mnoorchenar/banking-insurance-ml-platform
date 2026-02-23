# ──────────────────────────────────────────────────────────────
# Banking & Insurance ML Platform — Dockerfile
# HuggingFace Spaces compatible (port 7860, non-root user)
# ──────────────────────────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

# Create non-root user (required by HuggingFace Spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy project files
COPY --chown=user . .

EXPOSE 7860

# Preload models at startup, single worker (model is in-memory)
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "--preload", "app:app"]