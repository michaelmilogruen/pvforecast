#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for the improved LSTM model for low-resolution data.

This script loads a trained model and evaluates it on test data,
generating detailed analysis plots to assess model performance.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from datetime import datetime
import glob
import re

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def find_latest_model():
    """Find the latest model and associated scalers."""
    # Find the latest model file
    model_files = glob.glob('models/lstm_lowres_improved/final_model_*.keras')
    if not model_files:
        raise FileNotFoundError("No model files found in models/lstm_lowres_improved/")
    
    # Extract timestamps and find the latest
    timestamps = [re.search(r'final_model_(\d+_\d+).keras', f).group(1) for f in model_files]
    latest_timestamp = max(timestamps)
    
    latest_model = f'models/lstm_lowres_improved/final_model_{latest_timestamp}.keras'
    
    # Find corresponding scalers
    minmax_scaler = f'models/lstm_lowres_improved/minmax_scaler_{latest_timestamp}.pkl'
    standard_scaler = f'models/lstm_lowres_improved/standard_scaler_{latest_timestamp}.pkl'
    robust_scaler = f'models/lstm_lowres_improved/robust_scaler_{latest_timestamp}.pkl'
    target_scaler = f'models/lstm_lowres_improved/target_scaler_{latest_timestamp}.pkl'
    
    return {
        'model': latest_model,
        'minmax_scaler': minmax_scaler if os.path.exists(minmax_scaler) else None,
        'standard_scaler': standard_scaler if os.path.exists(standard_scaler) else None,
        'robust_scaler': robust_scaler if os.path.exists(robust_scaler) else None,
        'target_scaler': target_scaler if os.path.exists(target_scaler) else None,
        'timestamp': latest_timestamp
    }

def load_and_prepare_data(data_path):
    """Load and prepare data for testing."""
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"Data shape: {df.shape}")
    print(f"Data range: {df.index.min()} to {df.index.max()}")
    
    # Check for missing values
    missing_values = df.isna().sum()
    if missing_values.sum() > 0:
        print("Missing values in dataset:")
        print(missing_values[missing_values > 0])
        
        # Fill missing values
        print("Filling missing values...")
        df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df

def prepare_features(df):
    """Prepare features for the model."""
    # Define features to use
    features = [
        'GlobalRadiation [W m-2]',
        'Temperature [degree_Celsius]',
        'WindSpeed [m s-1]',
        'ClearSkyIndex',
        'hour_sin',
        'hour_cos',
        'day_sin',
        'day_cos',
        'isNight'
    ]
    
    # Verify all features exist in the dataframe
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        # Remove missing features
        features = [f for f in features if f in df.columns]
    
    # Group features by scaling method
    minmax_features = ['GlobalRadiation [W m-2]', 'ClearSkyIndex']
    standard_features = ['Temperature [degree_Celsius]']
    robust_features = ['WindSpeed [m s-1]']
    no_scaling_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'isNight']
    
    # Filter to only include features that exist in the dataframe
    minmax_features = [f for f in minmax_features if f in features]
    standard_features = [f for f in standard_features if f in features]
    robust_features = [f for f in robust_features if f in features]
    no_scaling_features = [f for f in no_scaling_features if f in features]
    
    print(f"MinMaxScaler features: {minmax_features}")
    print(f"StandardScaler features: {standard_features}")
    print(f"RobustScaler features: {robust_features}")
    print(f"No scaling features: {no_scaling_features}")
    
    target = 'power_w'
    
    return {
        'all_features': features,
        'minmax_features': minmax_features,
        'standard_features': standard_features,
        'robust_features': robust_features,
        'no_scaling_features': no_scaling_features,
        'target': target
    }

def split_and_scale_data(df, feature_info, model_files):
    """Split data and scale using pre-trained scalers."""
    # Split data into train, validation, and test sets
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size+val_size]
    test_df = df.iloc[train_size+val_size:]
    
    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")
    
    # Load pre-trained scalers
    minmax_scaler = joblib.load(model_files['minmax_scaler']) if model_files['minmax_scaler'] else None
    standard_scaler = joblib.load(model_files['standard_scaler']) if model_files['standard_scaler'] else None
    robust_scaler = joblib.load(model_files['robust_scaler']) if model_files['robust_scaler'] else None
    target_scaler = joblib.load(model_files['target_scaler'])
    
    # Function to scale features using appropriate scalers
    def scale_features(df_subset):
        scaled_data = {}
        
        # Apply MinMaxScaler to appropriate features
        if minmax_scaler and feature_info['minmax_features']:
            scaled_minmax = minmax_scaler.transform(df_subset[feature_info['minmax_features']])
            for i, feature in enumerate(feature_info['minmax_features']):
                scaled_data[feature] = scaled_minmax[:, i]
        
        # Apply StandardScaler to appropriate features
        if standard_scaler and feature_info['standard_features']:
            scaled_standard = standard_scaler.transform(df_subset[feature_info['standard_features']])
            for i, feature in enumerate(feature_info['standard_features']):
                scaled_data[feature] = scaled_standard[:, i]
        
        # Apply RobustScaler to appropriate features
        if robust_scaler and feature_info['robust_features']:
            scaled_robust = robust_scaler.transform(df_subset[feature_info['robust_features']])
            for i, feature in enumerate(feature_info['robust_features']):
                scaled_data[feature] = scaled_robust[:, i]
        
        # Add features that don't need scaling
        for feature in feature_info['no_scaling_features']:
            scaled_data[feature] = df_subset[feature].values
        
        # Create DataFrame with all scaled features
        scaled_df = pd.DataFrame(scaled_data, index=df_subset.index)
        
        # Ensure columns are in the same order as all_features
        return scaled_df[feature_info['all_features']]
    
    # Apply scaling to each dataset
    X_train = scale_features(train_df)
    X_val = scale_features(val_df)
    X_test = scale_features(test_df)
    
    # Scale target variable
    y_train = target_scaler.transform(train_df[[feature_info['target']]])
    y_val = target_scaler.transform(val_df[[feature_info['target']]])
    y_test = target_scaler.transform(test_df[[feature_info['target']]])
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'test_df': test_df,
        'target_scaler': target_scaler
    }

def create_sequences(X, y, time_steps):
    """Create sequences for time series forecasting."""
    X_seq, y_seq = [], []
    
    for i in range(len(X) - time_steps):
        X_seq.append(X.iloc[i:i + time_steps].values)
        y_seq.append(y[i + time_steps])
        
    return np.array(X_seq), np.array(y_seq)

def evaluate_model(model, X_test_seq, y_test, target_scaler):
    """Evaluate the model and return metrics."""
    # Make predictions
    y_pred = model.predict(X_test_seq)
    
    # Calculate metrics on scaled data
    mse_scaled = mean_squared_error(y_test, y_pred)
    rmse_scaled = np.sqrt(mse_scaled)
    mae_scaled = mean_absolute_error(y_test, y_pred)
    r2_scaled = r2_score(y_test, y_pred)
    
    print("\nModel Evaluation (Scaled):")
    print(f"Mean Squared Error (MSE): {mse_scaled:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse_scaled:.4f}")
    print(f"Mean Absolute Error (MAE): {mae_scaled:.4f}")
    print(f"R² Score: {r2_scaled:.4f}")
    
    # Inverse transform predictions and true values
    y_test_inv = target_scaler.inverse_transform(y_test)
    y_pred_inv = target_scaler.inverse_transform(y_pred)
    
    # Calculate metrics on original scale
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv, y_pred_inv)
    
    print("\nModel Evaluation (Original Scale):")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mse_scaled': mse_scaled,
        'rmse_scaled': rmse_scaled,
        'mae_scaled': mae_scaled,
        'r2_scaled': r2_scaled,
        'y_test_inv': y_test_inv,
        'y_pred_inv': y_pred_inv,
        'y_test': y_test,
        'y_pred': y_pred
    }

def plot_detailed_analysis(evaluation_results, test_df, timestamp):
    """Create detailed analysis plots for model evaluation."""
    # Create directory for analysis plots
    os.makedirs(f'results/lstm_lowres_improved_analysis', exist_ok=True)
    
    # 1. Scatter plot with regression line
    plt.figure(figsize=(10, 8))
    y_test = evaluation_results['y_test_inv'].flatten()
    y_pred = evaluation_results['y_pred_inv'].flatten()
    
    plt.scatter(y_test, y_pred, alpha=0.5)
    
    # Add perfect prediction line
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
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    plt.savefig(f'results/lstm_lowres_improved_analysis/scatter_plot_{timestamp}.png', dpi=300, bbox_inches='tight')
    
    # 2. Time series comparison (sample of test data)
    plt.figure(figsize=(16, 8))
    
    # Take a sample of 500 points or less
    sample_size = min(500, len(y_test))
    indices = np.arange(sample_size)
    
    plt.plot(indices, y_test[:sample_size], 'b-', label='Actual', linewidth=2)
    plt.plot(indices, y_pred[:sample_size], 'r-', label='Predicted', linewidth=2)
    
    plt.title('Time Series Comparison: Actual vs Predicted Power', fontsize=16)
    plt.xlabel('Time Steps', fontsize=14)
    plt.ylabel('Power Output (W)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    plt.savefig(f'results/lstm_lowres_improved_analysis/time_series_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
    
    # 3. Residual plot
    plt.figure(figsize=(10, 8))
    
    residuals = y_test - y_pred
    plt.scatter(y_test, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    
    plt.title('Residual Plot', fontsize=16)
    plt.xlabel('Actual Power Output (W)', fontsize=14)
    plt.ylabel('Residuals (Actual - Predicted)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f'results/lstm_lowres_improved_analysis/residual_plot_{timestamp}.png', dpi=300, bbox_inches='tight')
    
    # 4. Error distribution
    plt.figure(figsize=(10, 8))
    
    plt.hist(residuals, bins=50, alpha=0.75)
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
    
    plt.title('Error Distribution', fontsize=16)
    plt.xlabel('Prediction Error (W)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f'results/lstm_lowres_improved_analysis/error_distribution_{timestamp}.png', dpi=300, bbox_inches='tight')
    
    # 5. RMSE by power range
    plt.figure(figsize=(12, 8))
    
    # Create power ranges
    power_ranges = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 5000), (5000, float('inf'))]
    range_labels = ['0-1kW', '1-2kW', '2-3kW', '3-4kW', '4-5kW', '>5kW']
    rmse_by_range = []
    
    for power_range in power_ranges:
        mask = (y_test >= power_range[0]) & (y_test < power_range[1])
        if np.sum(mask) > 0:
            range_rmse = np.sqrt(mean_squared_error(y_test[mask], y_pred[mask]))
            rmse_by_range.append(range_rmse)
        else:
            rmse_by_range.append(0)
    
    plt.bar(range_labels, rmse_by_range)
    plt.title('RMSE by Power Range', fontsize=16)
    plt.xlabel('Power Range', fontsize=14)
    plt.ylabel('RMSE (W)', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(f'results/lstm_lowres_improved_analysis/rmse_by_power_range_{timestamp}.png', dpi=300, bbox_inches='tight')
    
    # 6. R² by power range
    plt.figure(figsize=(12, 8))
    
    r2_by_range = []
    for power_range in power_ranges:
        mask = (y_test >= power_range[0]) & (y_test < power_range[1])
        if np.sum(mask) > 10:  # Need enough points for meaningful R²
            range_r2 = r2_score(y_test[mask], y_pred[mask])
            r2_by_range.append(range_r2)
        else:
            r2_by_range.append(0)
    
    plt.bar(range_labels, r2_by_range)
    plt.title('R² Score by Power Range', fontsize=16)
    plt.xlabel('Power Range', fontsize=14)
    plt.ylabel('R² Score', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(f'results/lstm_lowres_improved_analysis/r2_by_power_range_{timestamp}.png', dpi=300, bbox_inches='tight')
    
    # 7. Hourly pattern analysis
    if 'hour_sin' in test_df.columns and 'hour_cos' in test_df.columns:
        plt.figure(figsize=(12, 8))
        
        # Extract hour from index
        test_hours = test_df.index.hour
        
        # Calculate average actual and predicted by hour
        hour_actual = []
        hour_pred = []
        
        for hour in range(24):
            hour_mask = test_hours[:-len(test_hours) + len(y_test)] == hour
            if np.sum(hour_mask) > 0:
                hour_actual.append(np.mean(y_test[hour_mask]))
                hour_pred.append(np.mean(y_pred[hour_mask]))
            else:
                hour_actual.append(0)
                hour_pred.append(0)
        
        plt.plot(range(24), hour_actual, 'b-', label='Actual', linewidth=2)
        plt.plot(range(24), hour_pred, 'r-', label='Predicted', linewidth=2)
        
        plt.title('Average Power Output by Hour', fontsize=16)
        plt.xlabel('Hour of Day', fontsize=14)
        plt.ylabel('Average Power Output (W)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.xticks(range(24))
        
        plt.savefig(f'results/lstm_lowres_improved_analysis/hourly_pattern_{timestamp}.png', dpi=300, bbox_inches='tight')
    
    print(f"Detailed analysis plots saved to results/lstm_lowres_improved_analysis/")

def main():
    """Main function to test the improved LSTM model."""
    # Find latest model and scalers
    try:
        model_files = find_latest_model()
        print(f"Loading model: {model_files['model']}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first using lstm_lowres_improved.py")
        return
    
    # Load model
    model = load_model(model_files['model'])
    
    # Load and prepare data
    df = load_and_prepare_data('data/station_data_1h.parquet')
    
    # Prepare features
    feature_info = prepare_features(df)
    
    # Split and scale data
    data = split_and_scale_data(df, feature_info, model_files)
    
    # Create sequences (use same sequence length as training)
    sequence_length = 48  # This should match the training sequence length
    X_test_seq, y_test_seq = create_sequences(data['X_test'], data['y_test'], sequence_length)
    
    print(f"Test sequences shape: {X_test_seq.shape}")
    
    # Evaluate model
    evaluation_results = evaluate_model(model, X_test_seq, y_test_seq, data['target_scaler'])
    
    # Plot detailed analysis
    plot_detailed_analysis(evaluation_results, data['test_df'], model_files['timestamp'])
    
    # Print final results
    print("\nFinal Results:")
    print(f"RMSE: {evaluation_results['rmse']:.2f}")
    print(f"MAE: {evaluation_results['mae']:.2f}")
    print(f"R²: {evaluation_results['r2']:.4f}")
    print(f"R² (Scaled): {evaluation_results['r2_scaled']:.4f}")

if __name__ == "__main__":
    main()