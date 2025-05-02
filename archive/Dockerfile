# Use TensorFlow's official image as base (includes Python and CUDA support)
FROM tensorflow/tensorflow:2.13.0-gpu

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsnappy-dev \
    python3-dev \
    python3-pip \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create necessary directories if they don't exist
RUN mkdir -p models/lstm logs results data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# Default command: run the forecast script
CMD ["python", "src/forecast.py"]

# Alternative entry point for model training
# To use: docker run --gpus all -it --rm your-image-name python src/run_lstm_models.py