# ==============================================================================
# AegisVision AI — Production Container Image
# Multi-stage optimized Dockerfile for YOLOv8 Computer Vision Platform
# ==============================================================================

FROM python:3.12-slim AS base

# Install system dependencies required for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/api/status || exit 1

# Launch FastAPI app with Uvicorn
CMD ["python", "app.py"]
