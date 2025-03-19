# -*- coding: utf-8 -*-
"""
Author: Michael Grün
Email: michaelgruen@hotmail.com
Description: This script loads a trained LSTM model and scalers to predict the power output of a photovoltaic system for the next day using features downloaded from an API.
Version: 1.0
Date: 2024-05-03
"""

import numpy as np
import pandas as pd
import joblib
import requests
import json
from datetime import datetime
import csv
from tensorflow.keras.models import load_model


# Global variables and constants

def fetch_weather_data():
    """
    Fetch weather forecast data from the API.

    Returns:
        dict: The fetched weather forecast data as a dictionary, or None if the request failed.
    """
    lat_lon = "47.38770748541585,15.094127778561258"
    current_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    fc_start_time = current_time.strftime("%Y-%m-%dT%H:%M")
    fc_end_time = (current_time + pd.Timedelta(hours=60)).strftime("%Y-%m-%dT%H:%M")  # Request 72 hours to ensure we have enough data

    url = f"https://dataset.api.hub.geosphere.at/v1/timeseries/forecast/nwp-v1-1h-2500m?lat_lon={lat_lon}&parameters=cape&parameters=cin&parameters=grad&parameters=mnt2m&parameters=mxt2m&parameters=rain_acc&parameters=rh2m&parameters=rr_acc&parameters=snow_acc&parameters=snowlmt&parameters=sp&parameters=sundur_acc&parameters=sy&parameters=t2m&parameters=tcc&parameters=u10m&parameters=ugust&parameters=v10m&parameters=vgust&start={fc_start_time}&end={fc_end_time}"

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        print("Weather data fetched successfully.")
        return data
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        return None
    
def predict_power(model, feature_scaler, target_scaler, new_data, seq_length=24):
    """
    Args:
        model: Trained LSTM model.
        feature_scaler: Fitted MinMaxScaler for the features.
        target_scaler: Fitted MinMaxScaler for the target.
        new_data: new input, can be either:
            1. A DataFrame with raw features that need to be scaled
            2. An already scaled numpy array (when pre-scaled features are provided)
        seq_length (int): Length of the input sequence for the LSTM model. Defaults to 24.

    Returns:
        numpy.ndarray: Inverse transformed predictions
    """
    # Check if new_data is already a numpy array (pre-scaled)
    if isinstance(new_data, np.ndarray):
        scaled_features = new_data
    else:
        # Otherwise, scale the features
        scaled_features = feature_scaler.transform(new_data)
    
    sequences = []
    for i in range(len(scaled_features) - seq_length + 1):
        sequences.append(scaled_features[i:(i + seq_length)])
    
    predictions = model.predict(np.array(sequences))
    return target_scaler.inverse_transform(predictions)

def process_weather_data(data):
    """
    Process the fetched weather forecast data and create a DataFrame.

    Args:
        data (dict): The fetched weather forecast data.

    Returns:
        pandas.DataFrame: A DataFrame containing the processed weather data with extracted features.
    """
    time_list = data['timestamps']
    parameters = data['features'][0]['properties']['parameters']

    # Calculate global irradiation difference and scale to J/(m²*s)
    global_irradiation = [(parameters['grad']['data'][i + 1] - val) / 3600 for i, val in enumerate(parameters['grad']['data'][:-1])]
    global_irradiation.insert(0, 0)  # Adding 0 as a placeholder

    # Create DataFrame with all parameters
    new_data = pd.DataFrame({
        'timestamp': pd.to_datetime(time_list),
        'temp_air': parameters['t2m']['data'],
        'wind_speed': np.abs(parameters['v10m']['data']),
        'poa_global': global_irradiation,  # Changed from 'global_irradiation' to 'poa_global'
        'cape': parameters['cape']['data'],
        'cin': parameters['cin']['data'],
        'min_temp': parameters['mnt2m']['data'],
        'max_temp': parameters['mxt2m']['data'],
        'rain_acc': parameters['rain_acc']['data'],
        'rel_humidity': parameters['rh2m']['data'],
        'total_precip': parameters['rr_acc']['data'],
        'snow_acc': parameters['snow_acc']['data'],
        'snow_limit': parameters['snowlmt']['data'],
        'surface_pressure': parameters['sp']['data'],
        'sunshine_duration': parameters['sundur_acc']['data'],
        'weather_symbol': parameters['sy']['data'],
        'total_cloud_cover': parameters['tcc']['data'],
        'wind_speed_east': parameters['u10m']['data'],
        'wind_gust_east': parameters['ugust']['data'],
        'wind_speed_north': parameters['v10m']['data'],
        'wind_gust_north': parameters['vgust']['data']
    })

    # Set timestamp as index
    new_data.set_index('timestamp', inplace=True)
    
    # Add derived features using the same approach as in training
    new_data = add_derived_features(new_data)
    
    # Reset index to keep timestamp as a column
    new_data.reset_index(inplace=True)

    return new_data

def add_derived_features(df):
    """
    Add derived time-based features consistent with the training process.
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        DataFrame with added time-based features
    """
    # Calculate hour with minute resolution (e.g., 15:15 becomes 15.25)
    df['hour'] = df.index.hour + df.index.minute / 60
    df['day_of_year'] = df.index.dayofyear
    
    # Add circular encoding for hour and day of year
    angle_hour = 2 * np.pi * df['hour'] / 24
    df['hour_sin'] = np.sin(angle_hour)
    df['hour_cos'] = np.cos(angle_hour)
    
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)
    
    # Add isNight feature (simplified approach - consider night if hour is outside 6-18)
    df['isNight'] = ((df['hour'] < 6) | (df['hour'] > 18)).astype(int)
    
    return df

def export_to_csv(data, filename='data/forecast_data.csv'):
    # Assuming 'data' is a list of dictionaries
    if not data:
        print("No data to export.")
        return

    # Extract field names from the first dictionary
    fieldnames = data[0].keys()

    # Write data to CSV
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"Data exported to {filename}")

def main():
    """
    Main function to fetch weather data, process it, and predict power output for the next day.
    """
    # Load model and scalers
    model = load_model('models/power_forecast_model.keras')
    feature_scaler = joblib.load('models/scaler_x.pkl')
    target_scaler = joblib.load('models/scaler_y.pkl')

    weather_data = fetch_weather_data()
    if weather_data is not None:
        new_data = process_weather_data(weather_data)
        
        # Create a new DataFrame with the same feature names as used during training
        # Map the inference feature names to the training feature names
        training_data = pd.DataFrame({
            'Station_GlobalRadiation [W m-2]': new_data['poa_global'],
            'Station_Temperature [degree_Celsius]': new_data['temp_air'],
            'Station_WindSpeed [m s-1]': new_data['wind_speed'],
            'Station_ClearSkyIndex': new_data['total_cloud_cover'],  # Using total cloud cover instead of default value
            'hour_sin': new_data['hour_sin'],
            'hour_cos': new_data['hour_cos'],
            'day_sin': new_data['day_sin'],
            'day_cos': new_data['day_cos'],
            'isNight': new_data['isNight']
        })
        
        # Define the features in the same order as used during training
        training_features = [
            'Station_GlobalRadiation [W m-2]',
            'Station_Temperature [degree_Celsius]',
            'Station_WindSpeed [m s-1]',
            'Station_ClearSkyIndex',
            'hour_sin',
            'hour_cos',
            'day_sin',
            'day_cos',
            'isNight'
        ]

        # Separate features that need scaling from those that don't (same approach as in training)
        time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'isNight']
        features_to_scale = [f for f in training_features if f not in time_features]
        
        # Scale only the features that need scaling
        scaled_features = feature_scaler.transform(training_data[features_to_scale])
        
        # Get the additional features that don't need scaling
        unscaled_features = training_data[time_features].values
        
        # Combine scaled and unscaled features
        X_combined = np.hstack((scaled_features, unscaled_features))
        
        # Use the predict_power function with the properly prepared features
        predicted_powers = predict_power(model, feature_scaler, target_scaler, X_combined)

        # Print the results to the console
        print("Predicted AC Power for the next 24 hours (in W):")
        print("| {:<20} | {:<15} | {:<15} | {:<15} | {:<25} | {:<15} | {:<15} | {:<15} | {:<15} | {:<15} | {:<15} | {:<15} | {:<15} | {:<15} | {:<15} | {:<15} | {:<15} | {:<15} | {:<15} | {:<15} | {:<15} |".format(
            'Time', 'Power (W)', 'Temp (°C)', 'Wind Speed (m/s)', 'Global Irr (J/m²s)',
            'CAPE (m²/s²)', 'CIN (J/kg)', 'Min Temp (°C)', 'Max Temp (°C)', 'Rain (kg/m²)',
            'Humidity (%)', 'Precip (kg/m²)', 'Snow (kg/m²)', 'Snow Lmt (m)', 'Press (Pa)',
            'Sun Dur (s)', 'Weather', 'Cloud Cover', 'Wind E (m/s)', 'Gust E (m/s)', 'Gust N (m/s)'))
        print("|" + "-" * 380 + "|")
        print("-" * 380)
        for i, power in enumerate(predicted_powers[:60]):  # Changed to 60 hours
            future_time = (datetime.now() + pd.Timedelta(hours=i)).strftime("%Y-%m-%d %H:%M")
            temp_air = new_data['temp_air'].iloc[i]
            wind_speed = new_data['wind_speed'].iloc[i]
            global_irradiation = new_data['poa_global'].iloc[i]
            
            
            print("|  {:<18} |  {:<15.2f} |  {:<15.2f} |  {:<15.2f} |  {:<25.2f} |  {:<15.2f} |  {:<15.2f} |  {:<15.2f} |  {:<15.2f} |  {:<15.2f} |  {:<15.2f} |  {:<15.2f} |  {:<15.2f} |  {:<15.2f} |  {:<15.2f} |  {:<15.2f} |  {:<15.2f} |  {:<15.2f} |  {:<15.2f} |  {:<15.2f} |  {:<15.2f} |".format(
                future_time, power[0], temp_air, wind_speed, global_irradiation,
                new_data['cape'].iloc[i], new_data['cin'].iloc[i], new_data['min_temp'].iloc[i], new_data['max_temp'].iloc[i],
                new_data['rain_acc'].iloc[i], new_data['rel_humidity'].iloc[i], new_data['total_precip'].iloc[i], new_data['snow_acc'].iloc[i],
                new_data['snow_limit'].iloc[i], new_data['surface_pressure'].iloc[i], new_data['sunshine_duration'].iloc[i], new_data['weather_symbol'].iloc[i],
                new_data['total_cloud_cover'].iloc[i], new_data['wind_speed_east'].iloc[i], new_data['wind_gust_east'].iloc[i], new_data['wind_gust_north'].iloc[i]))

        # Create a list of dictionaries for export
        export_data = []
        for i, power in enumerate(predicted_powers[:60]):  # Changed to 60 hours
            future_time = (datetime.now() + pd.Timedelta(hours=i)).strftime("%Y-%m-%d %H:%M")
            export_data.append({
                'timestamp': future_time,
                'power_w': max(0, power[0]),  # Ensure power is non-negative
                'temperature_c': new_data['temp_air'].iloc[i],
                'wind_speed_ms': new_data['wind_speed'].iloc[i],
                'global_irradiation': new_data['poa_global'].iloc[i],
                'cape': new_data['cape'].iloc[i],
                'cin': new_data['cin'].iloc[i],
                'min_temp': new_data['min_temp'].iloc[i],
                'max_temp': new_data['max_temp'].iloc[i],
                'rain': new_data['rain_acc'].iloc[i],
                'humidity': new_data['rel_humidity'].iloc[i],
                'precipitation': new_data['total_precip'].iloc[i],
                'snow': new_data['snow_acc'].iloc[i],
                'snow_limit': new_data['snow_limit'].iloc[i],
                'pressure': new_data['surface_pressure'].iloc[i],
                'sunshine_duration': new_data['sunshine_duration'].iloc[i],
                'weather_symbol': new_data['weather_symbol'].iloc[i],
                'cloud_cover': new_data['total_cloud_cover'].iloc[i],
                'wind_speed_east': new_data['wind_speed_east'].iloc[i],
                'wind_gust_east': new_data['wind_gust_east'].iloc[i],
                'wind_gust_north': new_data['wind_gust_north'].iloc[i]
            })

        # Export the data
        export_to_csv(export_data)

if __name__ == "__main__":
    main()
