import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt

# 1. Load and prepare the training data
def load_training_data(file_path):
    # Read CSV with first row as headers
    df = pd.read_csv(file_path)
    
    # Extract relevant features and target
    features = df[['temp_air', 'wind_speed', 'poa_global']]  # Using correct column names from your data
    target = df['AC Power']  # Using correct column name for power output
    
    return features, target

# 2. Create sequences for LSTM
def create_sequences(features, target, seq_length=24):
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:(i + seq_length)].values)
        y.append(target.iloc[i + seq_length])
    return np.array(X), np.array(y)

# 3. Build LSTM model
def build_model(seq_length, n_features, learning_rate=0.001):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(seq_length, n_features)),
        Dropout(0.2),
        LSTM(48, return_sequences=True),
        Dropout(0.2),
        LSTM(24, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                 loss='mse',
                 metrics=['mae'])
    return model

# Main execution
def main():
    # Load and prepare data
    features, target = load_training_data('merged_results.csv')
    
    # Scale the data
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    scaled_features = feature_scaler.fit_transform(features)
    scaled_target = target_scaler.fit_transform(target.values.reshape(-1, 1))
    
    # Create sequences
    seq_length = 96
    X, y = create_sequences(pd.DataFrame(scaled_features), pd.Series(scaled_target.flatten()), seq_length)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build model
    model = build_model(seq_length, X.shape[2])
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
    
    lr_reducer = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=4,
        min_lr=1e-6,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        'best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping, lr_reducer, model_checkpoint],
        verbose=1
    )
    
    # Load the best model
    model = tf.keras.models.load_model('best_model.keras')
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Inverse transform the scaled values
    y_test_orig = target_scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_orig = target_scaler.inverse_transform(y_pred)
    
    # Calculate metrics
    mse = mean_squared_error(y_test_orig, y_pred_orig)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_orig, y_pred_orig)
    r2 = r2_score(y_test_orig, y_pred_orig)
    
    print("\nTest Set Performance Metrics:")
    print(f"Root Mean Squared Error: {rmse:.2f} W")
    print(f"Mean Absolute Error: {mae:.2f} W")
    print(f"R² Score: {r2:.4f}")
    
    # Plot predictions vs actual
    plt.subplot(1, 2, 2)
    plt.scatter(y_test_orig, y_pred_orig, alpha=0.5)
    plt.plot([y_test_orig.min(), y_test_orig.max()], 
             [y_test_orig.min(), y_test_orig.max()], 
             'r--', lw=2)
    plt.xlabel('Actual Power (W)')
    plt.ylabel('Predicted Power (W)')
    plt.title('Predictions vs Actual Values')
    
    # Plot sample of time series prediction
    plt.figure(figsize=(15, 5))
    sample_size = 200
    plt.plot(y_test_orig[:sample_size], label='Actual', alpha=0.7)
    plt.plot(y_pred_orig[:sample_size], label='Predicted', alpha=0.7)
    plt.title('Power Output: Actual vs Predicted (Sample Period)')
    plt.xlabel('Time Steps')
    plt.ylabel('Power (W)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Save the model and scalers
    model.save('power_forecast_model.keras')
    
    return model, feature_scaler, target_scaler

# Function to make predictions with new data
def predict_power(model, feature_scaler, target_scaler, new_data, seq_length=24):
    """
    new_data should be a DataFrame with columns: temperature, wind_speed, global_irradiation
    """
    scaled_features = feature_scaler.transform(new_data)
    sequences = []
    for i in range(len(scaled_features) - seq_length):
        sequences.append(scaled_features[i:(i + seq_length)])
    
    predictions = model.predict(np.array(sequences))
    return target_scaler.inverse_transform(predictions)

if __name__ == "__main__":
    model, feature_scaler, target_scaler = main()
