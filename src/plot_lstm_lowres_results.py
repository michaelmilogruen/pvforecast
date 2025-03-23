#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot detailed results of the LSTM Low Resolution model on test data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import glob
from datetime import datetime

# Import the LSTMLowResForecaster class
from lstm_lowres import LSTMLowResForecaster

def load_latest_model():
    """Load the latest trained model based on timestamp in filename."""
    model_files = glob.glob('models/lstm_lowres/final_model_*.keras')
    if not model_files:
        raise FileNotFoundError("No trained model found in models/lstm_lowres/")
    
    # Sort by timestamp (newest first)
    latest_model_file = sorted(model_files)[-1]
    print(f"Loading model: {latest_model_file}")
    
    # Extract timestamp from filename
    timestamp = latest_model_file.split('_')[-1].split('.')[0]
    
    # Load scalers
    minmax_scaler_file = f'models/lstm_lowres/minmax_scaler_{timestamp}.pkl'
    standard_scaler_file = f'models/lstm_lowres/standard_scaler_{timestamp}.pkl'
    robust_scaler_file = f'models/lstm_lowres/robust_scaler_{timestamp}.pkl'
    
    scalers = {}
    if os.path.exists(minmax_scaler_file):
        scalers['minmax'] = joblib.load(minmax_scaler_file)
    if os.path.exists(standard_scaler_file):
        scalers['standard'] = joblib.load(standard_scaler_file)
    if os.path.exists(robust_scaler_file):
        scalers['robust'] = joblib.load(robust_scaler_file)
    
    # Load model
    model = tf.keras.models.load_model(latest_model_file)
    
    return model, scalers, timestamp

def prepare_test_data(forecaster, timestamp):
    """Prepare test data for prediction."""
    # Load data
    df = forecaster.load_and_prepare_data('data/station_data_1h.parquet')
    
    # Prepare features
    feature_info = forecaster.prepare_features(df)
    
    # Split data
    data = forecaster.split_and_scale_data(df, feature_info)
    
    # Create sequences for test data
    X_test_seq, y_test_seq = forecaster.create_sequences(
        data['X_test'], data['y_test'], forecaster.sequence_length
    )
    
    return X_test_seq, y_test_seq, df, data

def plot_detailed_results(model, X_test_seq, y_test, df, data, timestamp):
    """Plot detailed results of the model on test data."""
    # Make predictions
    y_pred = model.predict(X_test_seq)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Evaluation on Test Data:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Create output directory
    os.makedirs('results/lstm_lowres_analysis', exist_ok=True)
    
    # 1. Plot actual vs predicted values (time series)
    plt.figure(figsize=(16, 8))
    
    # Get a subset for better visualization
    sample_size = min(500, len(y_test))
    indices = np.arange(sample_size)
    
    plt.plot(indices, y_test[:sample_size], 'b-', label='Actual Power Output', linewidth=2)
    plt.plot(indices, y_pred[:sample_size], 'r-', label='Predicted Power Output', linewidth=2)
    plt.title('Actual vs Predicted PV Power (Test Data)', fontsize=16)
    plt.xlabel('Time Steps', fontsize=14)
    plt.ylabel('Power Output (W)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(f'results/lstm_lowres_analysis/time_series_comparison_{timestamp}.png', dpi=300)
    
    # 2. Scatter plot with regression line
    plt.figure(figsize=(12, 10))
    
    # Plot scatter points
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label='Predictions')
    
    # Plot perfect prediction line
    max_val = max(np.max(y_test), np.max(y_pred))
    min_val = min(np.min(y_test), np.min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction', linewidth=2)
    # Add regression line
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(y_test.reshape(-1, 1), y_pred)
    plt.plot(y_test, reg.predict(y_test.reshape(-1, 1)), 'g-',
             label=f'Regression Line (slope={float(reg.coef_):.4f})', linewidth=2)
    
    plt.title('Actual vs Predicted Power Output', fontsize=16)
    plt.xlabel('Actual Power Output (W)', fontsize=14)
    plt.ylabel('Predicted Power Output (W)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(f'results/lstm_lowres_analysis/scatter_plot_{timestamp}.png', dpi=300)
    
    # 3. Residual plot
    plt.figure(figsize=(12, 8))
    
    residuals = y_pred - y_test
    plt.scatter(y_test, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.title('Residual Plot', fontsize=16)
    plt.xlabel('Actual Power Output (W)', fontsize=14)
    plt.ylabel('Residuals (Predicted - Actual)', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(f'results/lstm_lowres_analysis/residual_plot_{timestamp}.png', dpi=300)
    
    # 4. Error distribution
    plt.figure(figsize=(12, 8))
    
    plt.hist(residuals, bins=50, alpha=0.75, color='blue')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
    plt.title('Error Distribution', fontsize=16)
    plt.xlabel('Prediction Error (W)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(f'results/lstm_lowres_analysis/error_distribution_{timestamp}.png', dpi=300)
    
    # 5. Daily pattern analysis
    # Extract hour from the test data index
    test_df = data['X_test'].iloc[forecaster.sequence_length:]
    test_df = test_df.iloc[:len(y_test)]  # Ensure same length as predictions
    
    # Add predictions and actual values to the dataframe
    test_df['actual'] = y_test
    test_df['predicted'] = y_pred
    
    # Group by hour and calculate mean
    if hasattr(test_df.index, 'hour'):
        hourly_avg = test_df.groupby(test_df.index.hour)[['actual', 'predicted']].mean()
        
        plt.figure(figsize=(14, 8))
        plt.plot(hourly_avg.index, hourly_avg['actual'], 'b-', label='Actual', linewidth=2)
        plt.plot(hourly_avg.index, hourly_avg['predicted'], 'r-', label='Predicted', linewidth=2)
        plt.title('Average Power Output by Hour of Day', fontsize=16)
        plt.xlabel('Hour of Day', fontsize=14)
        plt.ylabel('Average Power Output (W)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.xticks(range(0, 24))
        plt.tight_layout()
        
        plt.savefig(f'results/lstm_lowres_analysis/hourly_pattern_{timestamp}.png', dpi=300)
    
    # 6. Performance by power range
    plt.figure(figsize=(14, 8))
    
    # Create power bins
    power_bins = [0, 100, 500, 1000, 2000, 5000, 10000]
    bin_labels = ['0-100', '100-500', '500-1000', '1000-2000', '2000-5000', '5000+']
    
    # Digitize the actual power values into bins
    bin_indices = np.digitize(y_test.flatten(), power_bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_labels) - 1)
    
    # Calculate RMSE for each bin
    bin_rmse = []
    bin_counts = []
    
    for i in range(len(bin_labels)):
        mask = (bin_indices == i)
        if np.sum(mask) > 0:
            bin_rmse.append(np.sqrt(mean_squared_error(y_test[mask], y_pred[mask])))
            bin_counts.append(np.sum(mask))
        else:
            bin_rmse.append(0)
            bin_counts.append(0)
    
    # Plot RMSE by power range
    plt.bar(bin_labels, bin_rmse, alpha=0.7)
    plt.title('RMSE by Power Output Range', fontsize=16)
    plt.xlabel('Power Output Range (W)', fontsize=14)
    plt.ylabel('RMSE', fontsize=14)
    
    # Add count labels on top of bars
    for i, count in enumerate(bin_counts):
        plt.text(i, bin_rmse[i] + 10, f'n={count}', ha='center')
    
    plt.grid(True, axis='y')
    plt.tight_layout()
    
    plt.savefig(f'results/lstm_lowres_analysis/rmse_by_power_range_{timestamp}.png', dpi=300)
    
    # 7. R² by power range
    plt.figure(figsize=(14, 8))
    
    # Calculate R² for each bin
    bin_r2 = []
    
    for i in range(len(bin_labels)):
        mask = (bin_indices == i)
        if np.sum(mask) > 10:  # Need enough samples for meaningful R²
            bin_r2.append(r2_score(y_test[mask], y_pred[mask]))
        else:
            bin_r2.append(np.nan)  # Not enough samples
    
    # Plot R² by power range
    plt.bar(bin_labels, bin_r2, alpha=0.7)
    plt.title('R² Score by Power Output Range', fontsize=16)
    plt.xlabel('Power Output Range (W)', fontsize=14)
    plt.ylabel('R² Score', fontsize=14)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    
    # Add count labels on top of bars
    for i, count in enumerate(bin_counts):
        if not np.isnan(bin_r2[i]):
            plt.text(i, max(bin_r2[i] + 0.05, 0.05), f'n={count}', ha='center')
    
    plt.grid(True, axis='y')
    plt.tight_layout()
    
    plt.savefig(f'results/lstm_lowres_analysis/r2_by_power_range_{timestamp}.png', dpi=300)
    
    print(f"\nDetailed analysis plots saved to results/lstm_lowres_analysis/")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

if __name__ == "__main__":
    # Load the latest model
    model, scalers, timestamp = load_latest_model()
    
    # Initialize forecaster with the same parameters
    forecaster = LSTMLowResForecaster(
        sequence_length=24,
        batch_size=32,
        epochs=50
    )
    
    # Prepare test data
    X_test_seq, y_test, df, data = prepare_test_data(forecaster, timestamp)
    
    # Plot detailed results
    metrics = plot_detailed_results(model, X_test_seq, y_test, df, data, timestamp)
    
    print("\nFinal Results:")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"R²: {metrics['r2']:.4f}")