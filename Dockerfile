# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# Install system dependencies + dos2unix to sanitize line endings from Windows/Mac builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    dos2unix \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

# Default HF cache location inside the container.
# At runtime we prefer a network volume (/runpod-volume) when present so the
# ~24 GB FLUX weights survive cold starts.
ENV HF_HOME=/models/hf-cache
ENV TRANSFORMERS_CACHE=/models/hf-cache
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV PYTHONUNBUFFERED=1

RUN mkdir -p /models/hf-cache

# Copy only the application code (secrets and junk stay out via .dockerignore)
COPY src/ ./src/

# Convert any CRLF line endings to LF and ensure the handler is executable
RUN find /app -type f -name "*.py" -exec dos2unix {} + \
    && chmod +x /app/src/handler.py

# RunPod serverless entrypoint
CMD ["python3", "-u", "src/handler.py"]
