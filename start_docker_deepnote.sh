#!/bin/bash
# Script to attempt starting Docker daemon in Deepnote environment

echo "Attempting to start Docker daemon in Deepnote environment..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker daemon is running
if docker info &> /dev/null; then
    echo "Docker daemon is already running."
else
    echo "Docker daemon is not running. Attempting to start it..."
    
    # Try different methods to start Docker daemon
    
    # Method 1: Using service command
    echo "Trying to start Docker with service command..."
    sudo service docker start &> /dev/null
    
    # Method 2: Using dockerd directly
    if ! docker info &> /dev/null; then
        echo "Trying to start Docker daemon directly..."
        sudo dockerd > /tmp/dockerd.log 2>&1 &
        sleep 5
    fi
    
    # Check if Docker daemon is now running
    if docker info &> /dev/null; then
        echo "Docker daemon started successfully."
    else
        echo "Failed to start Docker daemon."
        echo "In Deepnote and some cloud environments, Docker daemon might not be available."
        echo "Consider using the direct installation method with deepnote_setup.sh instead."
        exit 1
    fi
fi

echo "Docker is ready to use. You can now run:"
echo "docker compose up -d"