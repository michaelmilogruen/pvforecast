# PV Forecast Docker Setup

This document provides instructions for running the PV Forecasting application using Docker.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- For GPU support: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## Quick Start

1. Build and start the container:

```bash
docker compose up -d
```

This will:
- Build the Docker image using the Dockerfile
- Start the container in detached mode
- Mount the data, models, results, and logs directories
- Run the forecast script by default

2. View logs:

```bash
docker compose logs -f
```

## Running Different Scripts

### To run the model training script:

```bash
docker compose run --rm pvforecast python src/run_lstm_models.py
```

### To run any other script:

```bash
docker compose run --rm pvforecast python src/your_script.py
```

## Building Without Docker Compose

If you prefer to use Docker directly:

1. Build the image:

```bash
docker build -t pvforecast .
```

2. Run the container:

```bash
docker run --gpus all -v ./data:/app/data -v ./models:/app/models -v ./results:/app/results -v ./logs:/app/logs pvforecast
```

## GPU Support

The Dockerfile and compose.yaml are configured to use GPU if available. Make sure you have:

1. NVIDIA GPU drivers installed on your host machine
2. NVIDIA Container Toolkit installed
3. Docker configured to use the NVIDIA runtime

## Customization

### Environment Variables

You can add environment variables in the `compose.yaml` file:

```yaml
environment:
  - TF_FORCE_GPU_ALLOW_GROWTH=true
  - PYTHONUNBUFFERED=1
  - YOUR_VARIABLE=value
```

### Volumes

The following volumes are mounted by default:
- `./data:/app/data`: For input and output data
- `./models:/app/models`: For trained models
- `./results:/app/results`: For visualization outputs
- `./logs:/app/logs`: For TensorBoard logs

## Troubleshooting

### GPU Issues

If you encounter GPU-related issues:

1. Verify GPU is visible to Docker:
```bash
docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

2. If using CPU only, modify the Dockerfile to use the CPU version of TensorFlow:
```dockerfile
FROM tensorflow/tensorflow:2.13.0
```

### Permission Issues

If you encounter permission issues with mounted volumes:

```bash
sudo chown -R $USER:$USER data models results logs