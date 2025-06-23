# AEGIS Security Co-Pilot - Optimized Multi-Stage Dockerfile
# Handles heavy ML packages separately to avoid timeouts

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# Set working directory
WORKDIR /app

# Install system dependencies (minimal for Cloud Run)
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Configure pip for better performance
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_RETRIES=3

# Upgrade pip and basic tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Stage 1: Install essential requirements first (fast layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Install heavy ML packages separately with better error handling
# Install PyTorch first (largest dependency)
RUN pip install --no-cache-dir \
    torch==2.3.1 \
    torchvision==0.18.1 \
    torchaudio==2.3.1 \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install Transformers and related (second largest)
RUN pip install --no-cache-dir \
    transformers==4.51.1 \
    tokenizers==0.21.1 \
    accelerate==1.7.0 \
    safetensors==0.5.3

# Install computer vision packages
RUN pip install --no-cache-dir \
    ultralytics==8.3.151 \
    supervision==0.25.1 \
    einops==0.8.1

# Install additional ML utilities (optional, can be removed if not needed)
RUN pip install --no-cache-dir \
    scikit-learn==1.7.0 \
    scipy==1.15.3 \
    onnxruntime==1.22.0

# Stage 3: Copy application code
COPY aegis/ ./aegis/
COPY main.py .
COPY setup.py .
COPY pyproject.yaml .

# Install the AEGIS package
RUN pip install --no-cache-dir -e .

# Create non-root user for security
RUN adduser --disabled-password --gecos "" aegisuser && \
    chown -R aegisuser:aegisuser /app

# Switch to non-root user
USER aegisuser

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV GOOGLE_GENAI_USE_VERTEXAI=True

# Create necessary directories
RUN mkdir -p /app/aegis/checkpoints

# Expose the port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:$PORT/list-apps || exit 1

# Start the application
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"] 