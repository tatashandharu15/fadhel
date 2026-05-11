# Use CUDA runtime base image
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    USE_TF=0 \
    USE_TORCH=1 \
    # Default generative model (CPU friendly, bilingual)
    DEFAULT_MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct" \
    # Default reranker model (DeBERTa v3 cross-encoder)
    RERANKER_MODEL_ID="cross-encoder/ms-marco-deberta-v3-base"

# Set working directory
WORKDIR /app

# Install system dependencies and Python runtime
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Ensure `python` points to Python 3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install PyTorch with CUDA support using wheels compatible with the host driver
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121

# Copy requirements
COPY requirements.txt .

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create a non-root user for security
RUN adduser --disabled-password --gecos '' appuser

# Create cache directory explicitly and set permissions
RUN mkdir -p /home/appuser/.cache/huggingface && \
    chown -R appuser:appuser /home/appuser

# Set working directory ownership
RUN chown -R appuser:appuser /app

USER appuser

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
