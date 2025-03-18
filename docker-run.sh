#!/bin/bash
# Script to build and run the PV Forecast Docker container

# Function to display help message
show_help() {
    echo "PV Forecast Docker Helper Script"
    echo ""
    echo "Usage: ./docker-run.sh [OPTION]"
    echo ""
    echo "Options:"
    echo "  -h, --help       Show this help message"
    echo "  -b, --build      Build the Docker image"
    echo "  -r, --run        Run the forecast script (default)"
    echo "  -t, --train      Run the model training script"
    echo "  -s, --shell      Start a shell in the container"
    echo "  -d, --down       Stop and remove containers"
    echo ""
    echo "Examples:"
    echo "  ./docker-run.sh -b -r    Build and run the forecast script"
    echo "  ./docker-run.sh -t       Run the model training script"
    echo "  ./docker-run.sh -s       Start a shell in the container"
}

# Default values
BUILD=false
RUN_FORECAST=false
RUN_TRAINING=false
RUN_SHELL=false
STOP_CONTAINERS=false

# Parse command line arguments
if [ $# -eq 0 ]; then
    # Default action if no arguments provided
    RUN_FORECAST=true
else
    while [ $# -gt 0 ]; do
        case "$1" in
            -h|--help)
                show_help
                exit 0
                ;;
            -b|--build)
                BUILD=true
                shift
                ;;
            -r|--run)
                RUN_FORECAST=true
                shift
                ;;
            -t|--train)
                RUN_TRAINING=true
                shift
                ;;
            -s|--shell)
                RUN_SHELL=true
                shift
                ;;
            -d|--down)
                STOP_CONTAINERS=true
                shift
                ;;
            *)
                echo "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
fi

# Stop and remove containers if requested
if [ "$STOP_CONTAINERS" = true ]; then
    echo "Stopping and removing containers..."
    docker-compose down
    exit 0
fi

# Build the Docker image if requested
if [ "$BUILD" = true ]; then
    echo "Building Docker image..."
    docker-compose build
fi

# Run the forecast script
if [ "$RUN_FORECAST" = true ]; then
    echo "Running forecast script..."
    docker-compose up
fi

# Run the model training script
if [ "$RUN_TRAINING" = true ]; then
    echo "Running model training script..."
    docker-compose run --rm pvforecast python src/run_lstm_models.py
fi

# Start a shell in the container
if [ "$RUN_SHELL" = true ]; then
    echo "Starting shell in container..."
    docker-compose run --rm pvforecast /bin/bash
fi

exit 0