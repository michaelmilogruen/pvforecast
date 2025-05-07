# -*- coding: utf-8 -*-
"""
Author: Michael Grün
Email: michaelgruen@hotmail.com
Description: This script calculates the power output of a photovoltaic system
             using an LSTM model and processes the results.
Version: 1.5 (Added SMAPE, R2 score, and post-processing clipping based on poa_global)
Date: 2024-05-07
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_curve
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import plot_model # Note: Requires pydot and graphviz to work

def split_sequences(X, y, n_steps):
    """
    Splits a multivariate dataset into sequences for time series forecasting.

    Args:
        X (np.ndarray): Feature dataset.
        y (np.ndarray): Target dataset.
        n_steps (int): The number of time steps in each sequence.

    Returns:
        tuple: A tuple containing the split feature sequences (X_) and
               corresponding target values (y_).
    """
    X_, y_ = list(), list()
    for i in range(len(X) - n_steps + 1):
        # Define the end of the input sequence
        end_ix = i + n_steps
        # Gather input and output parts of the sequence
        seq_x, seq_y = X[i:end_ix], y[end_ix - 1] # Predict the value at the end of the sequence
        X_.append(seq_x)
        y_.append(seq_y)
    return np.array(X_), np.array(y_)


def build_lstm_model(n_steps, n_features):
    """
    Builds a Sequential LSTM model for time series forecasting.

    Args:
        n_steps (int): The number of time steps in each input sequence.
        n_features (int): The number of features per time step.

    Returns:
        tf.keras.models.Sequential: The built LSTM model.
    """
    model = Sequential([
        Input(shape=(n_steps, n_features)),
        LSTM(64, activation='relu', return_sequences=True, dropout=0.4),
        LSTM(64, activation='relu', return_sequences=False, dropout=0.4),
        Dense(32, activation='relu'),
        Dense(1) # Output layer for regression
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.0005), metrics=['mse', 'mae'])
    return model


def plot_predictions(data, test_df_indices, predictions, y_test_unsc, n_steps):
    """
    Plots the actual PV power and the 1-day-ahead forecast.

    Args:
        data (pd.DataFrame): The original full dataset.
        test_df_indices (pd.Index): The original indices of the test set in 'data'.
        predictions (np.ndarray): The unscaled model predictions.
        y_test_unsc (np.ndarray): The unscaled actual target values for the test set.
        n_steps (int): The number of time steps used in the input sequences.
    """
    # The predictions correspond to the time steps starting n_steps - 1 after the start of the test set.
    # Get the original indices in 'data' that correspond to the predicted values.
    # The first prediction corresponds to the target at the end of the first sequence in the test set.
    # The indices in y_test correspond to the original indices in data starting from test_df_indices[n_steps - 1].
    start_index_in_test_df = n_steps - 1
    end_index_in_test_df = start_index_in_test_df + len(predictions) # len(predictions) is the number of sequences

    # Get the corresponding original indices from the full data DataFrame
    # These are the indices in the original 'data' DataFrame for the time steps being predicted.
    original_indices_for_predictions = test_df_indices[start_index_in_test_df : end_index_in_test_df]

    # Select the relevant rows from the original data using these indices
    # We need the original 'timestamp' and 'AC Power' for plotting.
    try:
        # Ensure 'timestamp' column exists after potential renaming in main()
        if 'timestamp' not in data.columns:
             print("Error: 'timestamp' column not found in data for plotting.")
             return

        pred_df = data.loc[original_indices_for_predictions][["timestamp", "AC Power"]].copy()
    except KeyError as e:
        print(f"Error accessing columns in original data for plotting: {e}")
        print("Available columns in data:", data.columns.tolist())
        return # Exit function if columns are not found

    # Add the predictions (which are already unscaled)
    # Ensure predictions are flat for assignment to a pandas Series
    pred_df.loc[:, "y_hat"] = predictions.flatten()

    # Convert timestamp to datetime and create 'RealTime' for plotting
    # Assuming the 'timestamp' column is in a format pandas can parse or the specified format
    try:
        pred_df["LocalDt"] = pd.to_datetime(pred_df["timestamp"]) # Let pandas infer format
        # Adjust 'RealTime' based on the original code's logic (assuming a timezone adjustment)
        # This adjustment might be specific to the user's data/timezone context
        # If the timestamp is already in the desired local time, this adjustment might not be needed.
        # Based on the original code, keeping it for consistency.
        pred_df["RealTime"] = pred_df["LocalDt"] - pd.Timedelta(hours=1)
    except Exception as e:
        print(f"Error converting timestamp column to datetime: {e}")
        print("Please check the format of your timestamp column.")
        return


    # Sort by time and set index for plotting
    pred_df = pred_df.sort_values(["RealTime"], ascending=True)
    pred_df = pred_df.set_index(pd.to_datetime(pred_df.RealTime), drop=True)

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.plot(pred_df.index, pred_df["y_hat"], color='blue', label="1-Day-Ahead PV Power Forecast (KW)")
    plt.plot(pred_df.index, pred_df["AC Power"], color='red', label="Actual PV Power (KW)")
    plt.ylabel('PV Power (KW)', fontsize=12)
    plt.title('Actual PV Power vs. 1-Day-Ahead Forecast')
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    plt.show()


def plot_roc_curve_regression(y_true, y_pred):
    """
    Generates and plots a ROC-like curve for regression results by
    considering different thresholds. This is not a standard ROC curve
    for classification but can provide insight into the model's ability
    to predict values above certain levels.

    Args:
        y_true (np.ndarray): The unscaled actual target values.
        y_pred (np.ndarray): The unscaled model predictions.
    """
    # Ensure inputs are flat arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Define a range of thresholds based on the actual values
    # Using a reasonable number of thresholds to get a curve
    if len(y_true) == 0:
        print("Cannot generate ROC-like curve: y_true is empty.")
        return

    # Use thresholds from min(true, pred) to max(true, pred) for better coverage
    min_val = min(y_true.min(), y_pred.min()) if y_pred.size > 0 else y_true.min()
    max_val = max(y_true.max(), y_pred.max()) if y_pred.size > 0 else y_true.max()

    if min_val == max_val:
         print("Cannot generate ROC-like curve: All values are the same.")
         return

    threshold_values = np.linspace(min_val, max_val, 100) # Increased thresholds for smoother curve

    tpr_values, fpr_values = [], []

    for threshold in threshold_values:
        # Convert regression problem to binary classification for this threshold
        # Class 1: value > threshold, Class 0: value <= threshold
        binary_true = (y_true > threshold).astype(int)
        binary_predictions = (y_pred > threshold).astype(int)

        # Calculate FPR and TPR using sklearn's roc_curve
        # roc_curve returns fpr, tpr, and thresholds. We need the values for the positive class (1).
        # Handle cases where binary_true might contain only one class
        if len(np.unique(binary_true)) < 2:
            # If only one class exists, roc_curve is not applicable for a curve
            # We can append a point if it makes sense (e.g., all true are 0, all pred are 0 -> TPR=0, FPR=0)
            # Or simply skip this threshold as it doesn't contribute to a curve
            continue

        fpr, tpr, _ = roc_curve(binary_true, binary_predictions)

        # roc_curve can return arrays of length > 2 if there are multiple thresholds in the data/predictions.
        # We are interested in the TPR and FPR for the positive class (index 1).
        # If the threshold results in only one class in binary_true or binary_predictions,
        # fpr or tpr might have length 1 or 2. Handle these cases.
        # Ensure tpr and fpr have at least 2 elements before accessing index 1
        if len(tpr) > 1 and len(fpr) > 1:
             tpr_values.append(tpr[1]) # TPR for positive class
             fpr_values.append(fpr[1]) # FPR for positive class
        elif len(tpr) == 1 and len(fpr) == 1:
             # Handle edge case where only one threshold is generated by roc_curve
             # This can happen if predictions are all above or all below the threshold
             # If binary_true contains both classes, this case shouldn't typically happen for meaningful data
             pass # Skip, as this point doesn't help define the curve shape
        # If len(tpr) or len(fpr) is 0, it means no samples or only one unique value. Skip.


    # Plotting the ROC-like curve
    plt.figure(figsize=(8, 6))
    # Ensure we only plot if we have valid TPR/FPR pairs
    if tpr_values and fpr_values:
        plt.plot(fpr_values, tpr_values, color='blue', lw=2, label='ROC-like Curve')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random Guess')
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('ROC-like Curve for Regression Thresholding')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("Could not generate ROC-like curve. Check thresholding or data distribution.")

def smape(y_true, y_pred):
    """
    Calculates the Symmetric Mean Absolute Percentage Error (SMAPE).

    Args:
        y_true (np.ndarray): The unscaled actual target values.
        y_pred (np.ndarray): The unscaled model predictions.

    Returns:
        float: The SMAPE value.
    """
    # Ensure inputs are flat arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Avoid division by zero when both actual and predicted are zero
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Use a small epsilon to prevent division by zero if denominator is zero
    # This happens when both y_true and y_pred are exactly 0.
    # In such cases, the error is 0, so the term for that data point is 0.
    # We can set the denominator to 1 where it's 0 to avoid NaN/inf, as the numerator will also be 0.
    denominator[denominator == 0] = 1

    return np.mean(np.abs(y_pred - y_true) / denominator) * 100


def main():
    # Modified data loading
    try:
        data = pd.read_csv("merged_results_modelchain.csv")
    except FileNotFoundError:
        print("Error: merged_results_modelchain.csv not found.")
        sys.exit(1)

    # --- Handle Timestamp Column ---
    timestamp_col_name = 'timestamp'
    if timestamp_col_name not in data.columns:
        # If 'timestamp' column is not found, check for 'Unnamed: 0'
        if 'Unnamed: 0' in data.columns:
            print("Warning: 'timestamp' column not found. Using 'Unnamed: 0' as timestamp.")
            data = data.rename(columns={'Unnamed: 0': timestamp_col_name})
        else:
            # If neither is found, raise an error
            print(f"Error: Neither '{timestamp_col_name}' nor 'Unnamed: 0' column found in the CSV file.")
            print("Available columns:", data.columns.tolist())
            sys.exit(1)
    # --- End Handle Timestamp Column ---

    # Ensure the timestamp column is in datetime format
    try:
        data[timestamp_col_name] = pd.to_datetime(data[timestamp_col_name])
    except Exception as e:
        print(f"Error converting timestamp column to datetime: {e}")
        print("Please check the format of your timestamp column.")
        sys.exit(1)


    # Define features and target
    features = [
        'poa_global',
        'temp_air',
        'wind_speed'
    ]
    target = 'AC Power'

    # Check if required columns exist in the data (excluding timestamp, which is handled above)
    required_cols_no_ts = features + [target]
    if not all(col in data.columns for col in required_cols_no_ts):
        missing = [col for col in required_cols_no_ts if col not in data.columns]
        print(f"Error: Missing required columns in the CSV file: {missing}")
        print("Available columns:", data.columns.tolist())
        sys.exit(1)


    # Add day/night indicator
    data['is_day'] = (data['poa_global'] > 0).astype(int)
    features.append('is_day')

    # --- Add time-related features ---
    # Extract hour and month for cyclical features
    data['hour'] = data[timestamp_col_name].dt.hour
    data['month'] = data[timestamp_col_name].dt.month

    # Add cyclical sine and cosine features for hour and month
    data['sin_hour'] = np.sin(2 * np.pi * data['hour'] / 24.0)
    data['cos_hour'] = np.cos(2 * np.pi * data['hour'] / 24.0)
    data['sin_month'] = np.sin(2 * np.pi * data['month'] / 12.0)
    data['cos_month'] = np.cos(2 * np.pi * data['month'] / 12.0)

    # Add these new features to the features list
    features.extend(['sin_hour', 'cos_hour', 'sin_month', 'cos_month'])
    # --- End Add time-related features ---


    # Define sequence length
    n_steps = 24 # Using 24 time steps for sequences (e.g., 24 hours)

    # Split data into training and testing sets
    train_size = int(len(data) * 0.8)
    train_df = data.iloc[:train_size].copy()
    test_df = data.iloc[train_size:].copy()

    # Correct scaling approach: Fit scalers only on the training data
    sc_x = MinMaxScaler().fit(train_df[features])
    sc_y = MinMaxScaler().fit(train_df[target].values.reshape(-1, 1))

    # Transform training data using the fitted scalers
    train_df[features] = sc_x.transform(train_df[features])
    train_df[target] = sc_y.transform(train_df[target].values.reshape(-1, 1))

    # Transform test data using the same fitted scalers
    test_df[features] = sc_x.transform(test_df[features])
    test_df[target] = sc_y.transform(test_df[target].values.reshape(-1, 1))

    # Print debugging information
    print("\nScaling Information:")
    print(f"Feature names: {features}")

    print("\nScaled Training Data Sample:")
    print("Features (first 5 rows):")
    print(train_df[features].head())
    print("\nTarget (first 5 rows):")
    print(train_df[target].head())

    print("\nOriginal vs Scaled Values (first 5 rows of training data):")
    # Get original values for comparison from the slice used for training
    original_train_target = data.iloc[:train_size][target]
    comparison_df = pd.DataFrame({
        'Original_Target': original_train_target.iloc[:5],
        'Scaled_Target': train_df[target].iloc[:5],
    })
    print(comparison_df)

    # Feature ranges after scaling
    print("\nFeature value ranges after scaling:")
    for feature in features:
        print(f"{feature}:")
        print(f"  Train min: {train_df[feature].min():.3f}, max: {train_df[feature].max():.3f}")
        print(f"  Test  min: {test_df[feature].min():.3f}, max: {test_df[feature].max():.3f}")

    # Target value ranges after scaling
    print("\nTarget value ranges after scaling:")
    print(f"Train min: {train_df[target].min():.3f}, max: {train_df[target].max():.3f}")
    print(f"Test  min: {test_df[target].min():.3f}, max: {test_df[target].max():.3f}")


    # Split scaled data into sequences
    x_train, y_train = split_sequences(train_df[features].values, train_df[target].values, n_steps)
    x_test, y_test = split_sequences(test_df[features].values, test_df[target].values, n_steps)

    # Print sequence shapes
    print("\nSequence shapes:")
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Build the LSTM model - Update n_features based on the new feature list
    model = build_lstm_model(x_train.shape[1], x_train.shape[2])

    # Print model summary
    model.summary()

    # Generate the model visualization (requires pydot and graphviz installed)
    try:
        plot_model(model,
                   to_file='model_architecture.png',
                   show_shapes=True,
                   show_dtype=False,
                   show_layer_names=True,
                   rankdir='TB',
                   dpi=96)
        print("\nModel architecture plot saved as model_architecture.png")
    except ImportError:
        print("\nCould not generate model architecture plot.")
        print("Please install pydot and graphviz (`pip install pydot graphviz`)")
        print("and ensure graphviz executables are in your system's PATH.")
    except Exception as e:
        print(f"\nAn error occurred while trying to plot the model: {e}")


    # Define callbacks for training
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    print("\nTraining the model...")
    history = model.fit(x_train, y_train,
                        validation_data=(x_test, y_test),
                        epochs=100, # Set a reasonable maximum number of epochs
                        batch_size=32,
                        callbacks=[checkpoint, early_stopping],
                        verbose=1)
    print("Model training finished.")

    # Load the best model weights
    try:
        model.load_weights('best_model.keras')
        print("Loaded best model weights.")
    except Exception as e:
        print(f"Could not load best model weights: {e}")
        print("Using the weights from the end of training.")


    # Make predictions on the test set
    print("\nMaking predictions on the test set...")
    predictions_scaled = model.predict(x_test)

    # Inverse transform predictions and actual values to original scale
    predictions_unsc = sc_y.inverse_transform(predictions_scaled.reshape(-1, 1))
    y_test_unsc = sc_y.inverse_transform(y_test.reshape(-1, 1))
    print("Predictions made and inverse transformed.")

    # --- Post-processing: Clip negative predictions to 0 ---
    predictions_unsc = np.maximum(0, predictions_unsc)
    print("Clipped negative predictions to 0.")
    # --- End clipping ---

    # --- Post-processing: Set predictions to 0 where poa_global is 0 ---
    # Get the poa_global values corresponding to the predicted time steps in the original test_df
    # These are the poa_global values at the end of each test sequence (n_steps - 1 index)
    poa_global_test = data.loc[test_df.index[n_steps - 1 : n_steps - 1 + len(predictions_unsc)], 'poa_global'].values.reshape(-1, 1)

    # Identify where poa_global is 0
    zero_poa_indices = poa_global_test == 0

    # Set predictions to 0 at these indices
    predictions_unsc[zero_poa_indices] = 0
    print("Set predictions to 0 where poa_global is 0.")
    # --- End post-processing ---


    # Evaluate the model
    mse = mean_squared_error(y_test_unsc, predictions_unsc)
    mae = mean_absolute_error(y_test_unsc, predictions_unsc)
    smape_value = smape(y_test_unsc, predictions_unsc) # Calculate SMAPE
    r2 = r2_score(y_test_unsc, predictions_unsc) # Calculate R2 score


    print(f"\nEvaluation Metrics (Unscaled):")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"SMAPE: {smape_value:.4f}%") # Print SMAPE
    print(f"R2 Score: {r2:.4f}") # Print R2 score


    # Calculate original MAPE, handling division by zero (kept for comparison if needed)
    non_zero_indices = y_test_unsc != 0
    if np.sum(non_zero_indices) > 0:
        # Add a small epsilon to the denominator to prevent division by near-zero values
        epsilon = 1e-8
        mape = np.mean(np.abs((y_test_unsc[non_zero_indices] - predictions_unsc[non_zero_indices]) / (y_test_unsc[non_zero_indices] + epsilon))) * 100
        print(f"Original MAPE (excluding zero actual values): {mape:.4f}%")
    else:
        print("Original MAPE could not be calculated as all actual values are zero.")


    print(f"Average value of y_test_unsc: {np.mean(y_test_unsc):.4f}")
    print(f"Number of epochs made: {len(history.epoch)}")


    # Plot the predictions against actual values
    print("\nGenerating prediction plot...")
    plot_predictions(data, test_df.index, predictions_unsc, y_test_unsc, n_steps)
    print("Prediction plot generated.")

    # Plot the ROC-like curve for regression thresholds
    print("\nGenerating ROC-like curve for regression...")
    plot_roc_curve_regression(y_test_unsc.flatten(), predictions_unsc.flatten())
    print("ROC-like curve generated.")


    # Create a DataFrame with predictions and actual values
    predictions_df = pd.DataFrame({
        "Actual (Unscaled)": y_test_unsc.flatten(),
        "Predicted (Unscaled)": predictions_unsc.flatten()
    })

    # Save predictions to a CSV file
    predictions_df.to_csv("predictions_vs_actual.csv", index=False)
    print("\nPredictions saved to predictions_vs_actual.csv")

    # Display the first few rows of the predictions DataFrame
    print("\nSample of Predictions vs Actual:")
    print(predictions_df.head())


if __name__ == "__main__":
    sys.exit(main())
