#!/bin/bash
# Setup script for running PV Forecast in Deepnote environment

echo "Setting up PV Forecast environment in Deepnote..."

# Install dependencies directly in the Deepnote environment
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories if they don't exist
mkdir -p models/lstm logs results

echo "Setup complete! You can now run the application with:"
echo "python src/forecast.py  # For forecasting"
echo "python src/run_lstm_models.py  # For model training"