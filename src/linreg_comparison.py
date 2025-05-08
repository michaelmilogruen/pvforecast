import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import math # Import math for pi
import joblib # Import joblib for saving the model
import matplotlib.pyplot as plt # Import matplotlib for plotting

def calculate_smape(y_true, y_pred):
    """
    Calculates the Symmetric Mean Absolute Percentage Error (SMAPE).
    Avoids division by zero by adding a small epsilon.
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    epsilon = 1e-8
    # Calculate the SMAPE formula
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred) + epsilon) # Add epsilon to avoid division by zero
    smape = np.mean(2.0 * numerator / denominator) * 100
    return smape

def calculate_mape(y_true, y_pred):
    """
    Calculates the Mean Absolute Percentage Error (MAPE).
    Avoids division by zero by adding a small epsilon.
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    epsilon = 1e-8
    # Calculate the MAPE formula
    # Using the standard MAPE formula: Sum(|y_true - y_pred| / |y_true|) * 100 / n
    # Handle division by zero for y_true=0
    abs_y_true = np.abs(y_true)
    # Create a mask for non-zero true values
    non_zero_mask = abs_y_true > epsilon
    # Calculate percentage error only for non-zero true values
    percentage_error = np.zeros_like(y_true, dtype=float) # Initialize with zeros
    percentage_error[non_zero_mask] = np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / abs_y_true[non_zero_mask])

    mape = np.mean(percentage_error) * 100
    return mape


def run_linear_regression_comparison(data_path='data/processed/station_data_10min.parquet', model_dir='model/'):
    """
    Runs a linear regression model using Global Radiation to predict PV Power,
    evaluates its performance, saves the trained model, and plots
    Actual vs. Predicted values (scatter and time series) for the test set.

    Args:
        data_path: Path to the 10-minute resolution data file.
        model_dir: Directory where the trained model will be saved.

    Returns:
        Dictionary of evaluation metrics for the linear regression model.
    """
    print(f"Running Linear Regression comparison using data from {data_path}")

    # --- Data Loading ---
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_parquet(data_path)
        print(f"Data shape: {df.shape}")

        # Ensure the index is a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            print("Warning: DataFrame index is not DatetimeIndex. Attempting to convert.")
            try:
                # Assuming the index column is named 'index' or similar, or is the first column
                # You might need to adjust this based on your parquet file structure
                if 'index' in df.columns:
                    df['index'] = pd.to_datetime(df['index'])
                    df = df.set_index('index')
                else:
                    # Attempt to use the current index if it's not DatetimeIndex
                    df.index = pd.to_datetime(df.index)
                print("Index converted to DatetimeIndex.")
            except Exception as e_index:
                print(f"Error converting index to datetime: {e_index}")
                print("Please ensure your data file has a proper datetime index or modify the load_data method.")
                return None # Exit if data file index conversion fails


        print(f"Data range: {df.index.min()} to {df.index.max()}")

        # Check for missing values
        missing_values = df.isna().sum()
        if missing_values.sum() > 0:
            print("Missing values in dataset:")
            print(missing_values[missing_values > 0])

            # Fill missing values using ffill followed by bfill for consistency
            print("Filling missing values using ffill followed by bfill...")
            df = df.fillna(method='ffill').fillna(method='bfill')
            print("Missing values after filling:", df.isna().sum().sum())
        else:
            print("No missing values found.")

    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return None # Exit if data file is not found
    except Exception as e:
        print(f"Error loading or processing data: {e}")
        return None # Exit on other data loading/processing errors

    # --- Feature and Target Selection ---
    feature_col = 'GlobalRadiation [W m-2]'
    target_col = 'power_w'

    if feature_col not in df.columns:
        print(f"Error: Feature column '{feature_col}' not found in the dataset. Exiting.")
        return None
    if target_col not in df.columns:
        print(f"Error: Target variable '{target_col}' not found in the dataset. Exiting.")
        return None

    # Select only the required columns
    df_selected = df[[feature_col, target_col]].copy()
    del df # Free up memory

    # --- Data Splitting ---
    # Use the same time-based split as the LSTM script
    df_selected = df_selected.sort_index() # Ensure data is sorted by time
    total_size = len(df_selected)
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.15)
    test_size = total_size - train_size - val_size # Use remaining for test

    train_df = df_selected.iloc[:train_size]
    val_df = df_selected.iloc[train_size : train_size + val_size]
    test_df = df_selected.iloc[train_size + val_size : ].copy() # Keep original test_df

    print(f"Total data size: {total_size}")
    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")

    # Prepare data for Linear Regression
    # Reshape X to be 2D (n_samples, n_features) as required by scikit-learn
    X_train = train_df[[feature_col]].values
    y_train = train_df[target_col].values

    X_test = test_df[[feature_col]].values
    y_test = test_df[target_col].values


    # --- Model Training ---
    print("\nTraining Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Training complete.")

    # --- Model Saving ---
    model_filename = 'linear_regression_model.joblib'
    save_path = os.path.join(model_dir, model_filename)

    print(f"\nAttempting to save the model to {save_path}...")
    try:
        # Create the model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        # Save the model using joblib
        joblib.dump(model, save_path)
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving the model: {e}")
        # Continue with evaluation even if saving fails

    # --- Prediction ---
    print("Making predictions on the test set...")
    y_pred_raw = model.predict(X_test)

    # --- Post-processing (Clip to non-negative and apply zero radiation rule) ---
    print("Applying post-processing...")
    y_pred_postprocessed = y_pred_raw.copy()

    # 1. Clip predictions to be non-negative (PV power cannot be negative)
    y_pred_postprocessed[y_pred_postprocessed < 0] = 0

    # 2. Set prediction to 0 if Global Radiation is 0 or very close
    RADIATION_THRESHOLD = 1.0 # W/m² - Use the same threshold as in the LSTM script
    zero_radiation_mask = (test_df[feature_col] < RADIATION_THRESHOLD).values # Get the mask from original test data
    y_pred_postprocessed[zero_radiation_mask] = 0

    print(f"Applied zero radiation post-processing to {zero_radiation_mask.sum()} predictions.")


    # --- Evaluation ---
    print("\nEvaluating Linear Regression model (Post-processed predictions):")

    # Ensure y_test and y_pred_postprocessed are 1D arrays for metric calculations
    y_test = y_test.flatten()
    y_pred_postprocessed = y_pred_postprocessed.flatten()

    mse = mean_squared_error(y_test, y_pred_postprocessed)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred_postprocessed)
    r2 = r2_score(y_test, y_pred_postprocessed)
    mape = calculate_mape(y_test, y_pred_postprocessed)
    smape = calculate_smape(y_test, y_pred_postprocessed)


    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"Symmetric Mean Absolute Percentage Error (SMAPE): {smape:.2f}%")
    print(f"R² Score: {r2:.4f}")

    evaluation_results = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'smape': smape,
        'r2': r2
    }

    # --- Plotting Actual vs. Predicted (Scatter Plot) ---
    print("\nGenerating Scatter plot of Actual vs. Predicted PV Power...")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_postprocessed, alpha=0.5, s=5) # s is marker size
    plt.title('Linear Regression: Actual vs. Predicted PV Power (Test Set)')
    plt.xlabel('Actual PV Power [W]')
    plt.ylabel('Predicted PV Power [W]')
    plt.grid(True)

    # Optional: Add a line representing perfect predictions
    max_val = max(y_test.max(), y_pred_postprocessed.max())
    plt.plot([0, max_val], [0, max_val], 'r--', lw=2) # Red dashed line

    plt.show() # Display the scatter plot


    # --- Plotting Actual vs. Predicted (Time Series Plot) ---
    print("\nGenerating Time Series plot of Actual vs. Predicted PV Power...")
    plt.figure(figsize=(15, 6)) # Wider figure for time series
    # Use the index from the test_df for the time axis
    plt.plot(test_df.index, y_test, label='Actual PV Power')
    plt.plot(test_df.index, y_pred_postprocessed, label='Predicted PV Power')

    plt.title('Linear Regression: Actual vs. Predicted PV Power Over Time (Test Set)')
    plt.xlabel('Time')
    plt.ylabel('PV Power [W]')
    plt.legend() # Show legend
    plt.grid(True)
    plt.tight_layout() # Adjust layout to prevent labels overlapping

    plt.show() # Display the time series plot


    return evaluation_results

if __name__ == "__main__":
    # Default data path (modify if your file is elsewhere)
    DATA_PATH = 'data/processed/station_data_10min.parquet'
    # Default model directory
    MODEL_DIRECTORY = 'model/'

    # Run the linear regression comparison and save the model
    metrics = run_linear_regression_comparison(data_path=DATA_PATH, model_dir=MODEL_DIRECTORY)

    if metrics:
        print("\n--- Linear Regression Final Results (Test Set) ---")
        print(f"RMSE: {metrics['rmse']:.2f}")
        print(f"MAE: {metrics['mae']:.2f}")
        print(f"MAPE: {metrics['mape']:.2f}%")
        print(f"SMAPE: {metrics['smape']:.2f}%")
        print(f"R²: {metrics['r2']:.4f}")
        print("-" * 45)
    else:
        print("\nLinear Regression comparison failed.")