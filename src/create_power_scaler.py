# -*- coding: utf-8 -*-
"""
Script to create a proper power scaler that maps from [0, 1] to the actual power range.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

def main():
    print("Creating power scaler to map from [0, 1] to actual power range...")
    
    # Create a scaler that maps from [0, 1] to [0, 5000]
    # 5000 W (5 kW) is a common size for residential PV systems
    # Adjust this value based on the actual system capacity
    max_power = 5000  # in watts
    
    # Create a MinMaxScaler that maps from [0, 1] to [0, max_power]
    power_scaler = MinMaxScaler(feature_range=(0, max_power))
    
    # Fit the scaler on [0, 1] range
    power_scaler.fit(np.array([[0], [1]]))
    
    # Save the scaler
    joblib.dump(power_scaler, 'models/power_scaler.pkl')
    
    print(f"Power scaler created and saved to models/power_scaler.pkl")
    print(f"This scaler maps from [0, 1] to [0, {max_power}] W")
    
    # Test the scaler
    test_values = np.array([[0], [0.25], [0.5], [0.75], [1]])
    scaled_values = power_scaler.transform(test_values)
    print("\nTest values:")
    for i, val in enumerate(test_values):
        print(f"  {val[0]:.2f} -> {scaled_values[i][0]:.2f} W")
    
    # Also test inverse transform
    inverse_values = power_scaler.inverse_transform(scaled_values)
    print("\nInverse transform test:")
    for i, val in enumerate(scaled_values):
        print(f"  {val[0]:.2f} W -> {inverse_values[i][0]:.2f}")
    
    return power_scaler

if __name__ == "__main__":
    main()