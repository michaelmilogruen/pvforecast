import os
import tensorflow as tf
from lstm_model import LSTMForecaster

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
os.makedirs('models/lstm', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Set parameters
SEQUENCE_LENGTH = 24  # Look back 24 time steps (6 hours with 15-min data)
BATCH_SIZE = 64
EPOCHS = 100

print(f"Running LSTM models with parameters:")
print(f"- Sequence length: {SEQUENCE_LENGTH}")
print(f"- Batch size: {BATCH_SIZE}")
print(f"- Max epochs: {EPOCHS}")

# Initialize forecaster
forecaster = LSTMForecaster(
    sequence_length=SEQUENCE_LENGTH,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS
)

# Run pipeline
metrics = forecaster.run_pipeline('data/processed_training_data.parquet')

# Print final comparison
print("\nModel Comparison:")
for set_name, metrics in metrics.items():
    print(f"{set_name} model - RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.4f}")