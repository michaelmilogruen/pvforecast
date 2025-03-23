#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for the LSTM Low Resolution model.
This script runs the LSTM model on the 1-hour resolution data.
"""

import os
import tensorflow as tf
from lstm_lowres import LSTMLowResForecaster

# Check for GPU availability
print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Set memory growth to avoid memory allocation issues
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth set to True for all GPUs")
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")

# Create directories if they don't exist
os.makedirs('models/lstm_lowres', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Set parameters
SEQUENCE_LENGTH = 24  # Look back 24 hours (1 day with 1-hour data)
BATCH_SIZE = 32
EPOCHS = 50

print(f"Running LSTM Low Resolution model with parameters:")
print(f"- Sequence length: {SEQUENCE_LENGTH}")
print(f"- Batch size: {BATCH_SIZE}")
print(f"- Max epochs: {EPOCHS}")

# Initialize forecaster
forecaster = LSTMLowResForecaster(
    sequence_length=SEQUENCE_LENGTH,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS
)

# Run pipeline
metrics = forecaster.run_pipeline()

# Print final results
print("\nFinal Results:")
print(f"RMSE: {metrics['rmse']:.2f}")
print(f"MAE: {metrics['mae']:.2f}")
print(f"R²: {metrics['r2']:.4f}")