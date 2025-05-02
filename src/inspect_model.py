import tensorflow as tf
from tensorflow import keras
import os

def inspect_keras_model(model_path):
    """
    Load and print the structure of a Keras model.
    
    Args:
        model_path: Path to the model file (.h5, .keras) or SavedModel directory
    """
    # Load the model
    try:
        model = keras.models.load_model(model_path)
        print("Model loaded successfully!")
        
        # Print model summary
        print("\nModel Summary:")
        model.summary()
        
        # Get layer configurations
        print("\nDetailed Layer Information:")
        for i, layer in enumerate(model.layers):
            print(f"Layer {i}: {layer.name}")
            print(f"  Type: {layer.__class__.__name__}")
            print(f"  Config: {layer.get_config()}")
            print(f"  Input shape: {layer.input_shape}")
            print(f"  Output shape: {layer.output_shape}")
            print("  -----------------------")
        
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

if __name__ == "__main__":
    # Replace with your model path
    model_path = "models/lstm/station_model.keras"  # or "path/to/your/saved_model"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
    else:
        model = inspect_keras_model(model_path)
