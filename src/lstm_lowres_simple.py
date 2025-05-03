import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import math # Import math for pi

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- Configuration ---
DATA_PATH = 'data/processed/station_data_1h.parquet' # Path to your data file
SEQUENCE_LENGTH = 24 # Number of time steps to look back (24 hours for 1-hour data)
BATCH_SIZE = 32
EPOCHS = 50 # Maximum number of epochs
VALIDATION_SPLIT = 0.15 # Percentage of data for validation
TEST_SPLIT = 0.15 # Percentage of data for testing

# Define the features to use (must match columns in your data)
# Includes features from the original script's prepare_features and create_time_features
FEATURES = [
    'GlobalRadiation [W m-2]',
    'ClearSkyDHI',
    'ClearSkyGHI',
    'ClearSkyDNI',
    'SolarZenith [degrees]',
    'AOI [degrees]',
    'isNight', # Assuming this is calculated or available
    'ClearSkyIndex',
    'hour_cos', # Assuming this is available in the raw data
    'Temperature [degree_Celsius]',
    'WindSpeed [m s-1]',
    # Engineered day features (will be added if not present)
    'day_sin',
    'day_cos',
]
TARGET = 'power_w' # Target variable column name

# --- Data Loading and Preparation ---

def load_data(data_path):
    """
    Load data from the specified file path and perform basic cleaning.
    """
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_parquet(data_path)
        print(f"Data shape: {df.shape}")

        # Ensure the index is a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            print("Warning: DataFrame index is not DatetimeIndex. Attempting to convert.")
            try:
                # Assuming the index column is named 'index' or similar, or is the first column
                if 'index' in df.columns:
                    df['index'] = pd.to_datetime(df['index'])
                    df = df.set_index('index')
                else:
                    df.index = pd.to_datetime(df.index)
                print("Index converted to DatetimeIndex.")
            except Exception as e_index:
                print(f"Error converting index to datetime: {e_index}")
                print("Please ensure your data file has a proper datetime index or modify the load_data method.")
                return None # Return None if conversion fails

        print(f"Data range: {df.index.min()} to {df.index.max()}")

        # Check for missing values and fill them
        missing_values = df.isna().sum()
        if missing_values.sum() > 0:
            print("Missing values found. Filling using ffill followed by bfill...")
            df = df.fillna(method='ffill').fillna(method='bfill')
            print("Missing values after filling:", df.isna().sum().sum())
        else:
            print("No missing values found.")

    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return None # Return None if file not found
    except Exception as e:
        print(f"Error loading or processing data: {e}")
        return None # Return None for other loading errors

    return df

def create_time_features(df):
    """
    Create simple time-based and cyclical features.
    'hour_cos' is expected to be in the raw data.
    'isNight' is calculated based on Global Radiation if available.
    """
    print("Creating simple time-based and cyclical features...")

    # Cyclical features for the day of the year
    df['day_sin'] = np.sin(2 * math.pi * df.index.dayofyear / 365.0)
    df['day_cos'] = np.cos(2 * math.pi * df.index.dayofyear / 365.0)

    # isNight calculation based on Global Radiation
    radiation_col_name = 'GlobalRadiation [W m-2]'
    if radiation_col_name in df.columns:
        RADIATION_THRESHOLD = 1.0 # W/m² - Adjust this threshold as needed
        df['isNight'] = (df[radiation_col_name] < RADIATION_THRESHOLD).astype(int)
        print("'isNight' feature created based on Global Radiation.")
    else:
        print(f"Warning: '{radiation_col_name}' not found. Cannot create 'isNight'. Adding placeholder.")
        df['isNight'] = 0 # Add a placeholder if radiation data is missing

    return df


def prepare_data(df, features, target, validation_split, test_split):
    """
    Selects features and target, splits data by time, and scales.
    Uses a single MinMaxScaler for all features and the target.
    """
    print("\nPreparing data: selecting features, splitting, and scaling...")

    # Verify all required features and target exist
    required_cols = features + [target]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: The following required columns are missing: {missing_cols}. Exiting.")
        return None, None, None, None, None # Return None if columns are missing

    # Select only the required columns
    df_processed = df[required_cols].copy()

    # Split data into train, validation, and test sets by time
    df_processed = df_processed.sort_index() # Ensure data is sorted by time
    total_size = len(df_processed)
    test_size = int(total_size * test_split)
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size - test_size

    train_df = df_processed.iloc[:train_size]
    val_df = df_processed.iloc[train_size : train_size + val_size]
    test_df = df_processed.iloc[train_size + val_size : ]

    print(f"Total data size: {total_size}")
    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")

    # Initialize and fit a single MinMaxScaler on training data for all columns
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Fit on all columns (features + target) of the training data
    scaler.fit(train_df[features + [target]])

    # Transform all datasets
    train_scaled = scaler.transform(train_df[features + [target]])
    val_scaled = scaler.transform(val_df[features + [target]])
    test_scaled = scaler.transform(test_df[features + [target]])

    # Separate features (X) and target (y) for each set
    X_train_scaled = train_scaled[:, :-1] # All columns except the last one (target)
    y_train_scaled = train_scaled[:, -1].reshape(-1, 1) # The last column (target)

    X_val_scaled = val_scaled[:, :-1]
    y_val_scaled = val_scaled[:, -1].reshape(-1, 1)

    X_test_scaled = test_scaled[:, :-1]
    y_test_scaled = test_scaled[:, -1].reshape(-1, 1)

    # Return original test_df for inverse transformation and plotting later
    return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled, scaler, test_df # Added test_df

def create_sequences(X, y, time_steps):
    """
    Create sequences for time series forecasting.
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        # Select the sequence of features ending at i + time_steps - 1
        X_seq.append(X[i : (i + time_steps)])
        # Select the target value at i + time_steps (the next step)
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

# --- Model Building ---

def build_simple_lstm_model(input_shape):
    """
    Builds a very simple Sequential LSTM model with one layer.
    """
    print("\nBuilding simple single-layer LSTM model...")
    model = Sequential()
    # Add a single LSTM layer
    # units: Number of LSTM units (neurons) - a basic starting point
    # input_shape: (sequence_length, number_of_features)
    # return_sequences=False because we are predicting a single value (the next step)
    model.add(LSTM(units=50, input_shape=input_shape, return_sequences=False))

    # Add a single Dense output layer
    # units=1 for predicting a single value (power output)
    model.add(Dense(units=1))

    # Compile model
    # optimizer: Adam is a common choice
    # loss: Mean Squared Error (MSE) is standard for regression
    # metrics: Mean Absolute Error (MAE) is easy to interpret
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    model.summary()
    return model

# --- Training ---

def train_model(model, X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs, batch_size):
    """
    Trains the LSTM model.
    """
    print("\nTraining model...")
    history = model.fit(
        X_train_seq, y_train_seq,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val_seq, y_val_seq),
        verbose=1 # Show training progress
    )
    return history

# --- Evaluation ---

def evaluate_model(model, X_test_seq, y_test_scaled, scaler, original_test_df):
    """
    Evaluates the model on the test set and calculates basic metrics.
    Inverse transforms predictions and actual values for metrics on original scale.
    """
    print("\nEvaluating model...")
    # Get scaled predictions
    y_pred_scaled = model.predict(X_test_seq)

    # Inverse transform actual and predicted values to original scale
    # The scaler was fitted on all columns (features + target).
    # To inverse transform the target (last column), we need to create dummy feature columns.
    # The shape of y_test_scaled and y_pred_scaled is (n_samples, 1).
    # We need to create arrays of shape (n_samples, num_features + 1) for inverse_transform.
    num_features = X_test_seq.shape[2] # Number of features
    dummy_features = np.zeros((len(y_test_scaled), num_features))

    # Combine dummy features with scaled target/predictions
    y_test_combined = np.hstack((dummy_features, y_test_scaled))
    y_pred_combined = np.hstack((dummy_features, y_pred_scaled))

    # Inverse transform the combined arrays
    y_test_inv = scaler.inverse_transform(y_test_combined)[:, -1] # Get the last column (target)
    y_pred_inv = scaler.inverse_transform(y_pred_combined)[:, -1] # Get the last column (target)

    # Ensure y_test_inv and y_pred_inv are 1D arrays for metrics calculation
    y_test_inv = y_test_inv.flatten()
    y_pred_inv = y_pred_inv.flatten()


    # Calculate metrics on the original scale
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv, y_pred_inv)

    # Note: MAPE and SMAPE can be tricky with zero actual values.
    # We'll calculate simple versions, adding a small epsilon to avoid division by zero.
    epsilon = 1e-8
    mape = np.mean(np.abs((y_test_inv - y_pred_inv) / (y_test_inv + epsilon))) * 100
    smape = np.mean(2.0 * np.abs(y_pred_inv - y_test_inv) / (np.abs(y_pred_inv) + np.abs(y_test_inv) + epsilon)) * 100


    print("\nEvaluation Metrics (Original Scale):")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"Symmetric Mean Absolute Percentage Error (SMAPE): {smape:.2f}%")
    print(f"R² Score: {r2:.4f}")

    # Return metrics and inverse transformed values for plotting
    return {
        'mse': mse, 'rmse': rmse, 'mae': mae, 'mape': mape, 'smape': smape, 'r2': r2,
        'y_test_inv': y_test_inv, 'y_pred_inv': y_pred_inv
    }

# --- Plotting ---

def plot_loss(history):
    """
    Plots training and validation loss from the training history.
    """
    print("\nPlotting training and validation loss...")
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show() # Display the plot

def plot_predictions(y_test_inv, y_pred_inv, sequence_length, original_test_df):
    """
    Plots a sample of actual vs predicted values on the original scale.
    Uses the index from the original test_df for plotting time series.
    """
    print("\nPlotting sample of actual vs predicted power...")

    # The predictions correspond to the timestamps starting after the sequence length
    # in the original test DataFrame.
    # The length of y_test_inv and y_pred_inv should be len(original_test_df) - sequence_length.
    # We need to get the index slice from original_test_df that matches these predictions.
    prediction_index = original_test_df.index[sequence_length:]

    # Ensure the index length matches the prediction length
    if len(prediction_index) != len(y_test_inv):
         print(f"Warning: Index length ({len(prediction_index)}) does not match prediction length ({len(y_test_inv)}). Cannot plot predictions.")
         return

    # Plot a sample (e.g., the first 1000 points)
    sample_size = min(1000, len(y_test_inv))
    sample_index = prediction_index[:sample_size]

    plt.figure(figsize=(14, 7))
    plt.plot(sample_index, y_test_inv[:sample_size], label='Actual Power Output (W)')
    plt.plot(sample_index, y_pred_inv[:sample_size], label='Predicted Power Output (W)', alpha=0.7)
    plt.title(f'Actual vs Predicted PV Power (W) - Sample ({sample_size} points)')
    plt.xlabel('Time')
    plt.ylabel('Power Output (W)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45) # Rotate x-axis labels for better readability
    plt.tight_layout() # Adjust layout
    plt.show() # Display the plot


# --- Main Execution ---

if __name__ == "__main__":
    print("Starting simple LSTM PV forecasting script...")

    # Check for GPU availability
    print("TensorFlow version:", tf.__version__)
    gpu_devices = tf.config.list_physical_devices('GPU')
    print("GPU Available:", "Yes" if gpu_devices else "No")
    if not gpu_devices:
        print("Warning: No GPU devices found. Training may be slow on CPU.")
    else:
         try:
            gpus = tf.config.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory growth set to True for all GPUs")
         except RuntimeError as e:
            print(f"Error setting memory growth: {e}")


    # Load data
    df = load_data(DATA_PATH)
    if df is None:
        exit() # Exit if data loading failed

    # Create time features
    df = create_time_features(df)

    # Prepare data: split, scale, and get original test_df
    (X_train_scaled, y_train_scaled,
     X_val_scaled, y_val_scaled,
     X_test_scaled, y_test_scaled,
     scaler, original_test_df) = prepare_data(df, FEATURES, TARGET, VALIDATION_SPLIT, TEST_SPLIT)

    if X_train_scaled is None:
        exit() # Exit if data preparation failed

    # Create sequences
    print(f"\nCreating sequences with sequence length: {SEQUENCE_LENGTH} steps ({SEQUENCE_LENGTH*1:.2f} hours lookback)")

    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, SEQUENCE_LENGTH)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, SEQUENCE_LENGTH)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, SEQUENCE_LENGTH) # Use y_test_scaled here for creating sequences

    print(f"Training sequences shape: {X_train_seq.shape}")
    print(f"Validation sequences shape: {X_val_seq.shape}")
    print(f"Testing sequences shape: {X_test_seq.shape}")

    # Build 
    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
    model = build_simple_lstm_model(input_shape)

    # Train model
    history = train_model(model, X_train_seq, y_train_seq, X_val_seq, y_val_seq, EPOCHS, BATCH_SIZE)

    # Evaluate model
    # Pass y_test_scaled (the scaled true targets for evaluation) and the original_test_df
    evaluation_results = evaluate_model(model, X_test_seq, y_test_seq, scaler, original_test_df)


    # Plot results
    plot_loss(history)
    # Pass the inverse transformed values and original_test_df for plotting
    plot_predictions(evaluation_results['y_test_inv'], evaluation_results['y_pred_inv'], SEQUENCE_LENGTH, original_test_df)

    print("\nSimple LSTM script finished.")
