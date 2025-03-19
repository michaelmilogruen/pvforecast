# -*- coding: utf-8 -*-
"""
Script to export feature and target scalers for the PV forecasting model.
This script only exports the scalers without training the model.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

def main():
    print("Loading processed training data...")
    # Load the processed training data
    data_path = 'data/processed_training_data.parquet'
    df = pd.read_parquet(data_path)
    
    print(f"Loaded data shape: {df.shape}")
    
    # Define feature sets (same as in lstm_model.py)
    feature_sets = {
        'station': [
            'Station_GlobalRadiation [W m-2]',
            'Station_Temperature [degree_Celsius]',
            'Station_WindSpeed [m s-1]',
            'Station_ClearSkyIndex',
            'hour_sin',
            'hour_cos',
            'day_cos',
            'day_sin',
            'isNight'
        ]
    }
    
    # Select the feature set to use (default to 'station')
    feature_set_name = 'station'
    features = feature_sets[feature_set_name]
    
    # Separate features that need scaling from those that don't
    time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'isNight']
    features_to_scale = [f for f in features if f not in time_features]
    additional_features = [f for f in features if f in time_features]
    
    print(f"Using feature set: {feature_set_name}")
    print(f"Features to scale: {features_to_scale}")
    print(f"Additional features: {additional_features}")
    
    target = 'power_w'  # Target variable (power output in watts)
    
    # Check for missing values
    missing_values = df[features + [target]].isna().sum()
    if missing_values.sum() > 0:
        print("Warning: Missing values in dataset:")
        print(missing_values[missing_values > 0])
    else:
        print("No missing values in the dataset.")
    
    # Create and fit the scalers
    print("Creating and fitting scalers...")
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    # Fit the feature scaler on features that need scaling
    scaler_x.fit(df[features_to_scale])
    
    # Fit the target scaler
    scaler_y.fit(df[[target]])
    
    # Save the scalers
    print("Saving scalers to models directory...")
    joblib.dump(scaler_x, 'models/scaler_x.pkl')
    joblib.dump(scaler_y, 'models/scaler_y.pkl')
    
    print("Scalers exported successfully.")
    
    # Print some information about the scalers
    print("\nScaler Information:")
    print(f"Feature scaler (X) - min: {scaler_x.data_min_}, max: {scaler_x.data_max_}")
    print(f"Target scaler (y) - min: {scaler_y.data_min_[0]}, max: {scaler_y.data_max_[0]}")
    
    # Print sample transformation
    print("\nSample Transformation:")
    # Sample feature values
    sample_features = df[features_to_scale].iloc[0:1]
    print(f"Original features: {sample_features.values[0]}")
    print(f"Scaled features: {scaler_x.transform(sample_features)[0]}")
    
    # Sample target value
    sample_target = df[[target]].iloc[0:1]
    print(f"Original target: {sample_target.values[0][0]}")
    print(f"Scaled target: {scaler_y.transform(sample_target)[0][0]}")
    
    return scaler_x, scaler_y

if __name__ == "__main__":
    main()