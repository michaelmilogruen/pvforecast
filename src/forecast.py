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
from lstma import predict_power

# Load the saved model and scalers
model = load_model('models/final_model.h5', compile=False)
sc_x = joblib.load('models/scaler_x.pkl')
sc_y = joblib.load('models/scaler_y.pkl')

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

    # Extract hour and month features
    new_data['hour'] = new_data['timestamp'].dt.hour
    new_data['sin_hour'] = np.sin(2 * np.pi * new_data['hour'] / 24.0)
    new_data['cos_hour'] = np.cos(2 * np.pi * new_data['hour'] / 24.0)
    new_data['month'] = new_data['timestamp'].dt.month
    new_data['sin_month'] = np.sin(2 * np.pi * new_data['month'] / 12.0)
    new_data['cos_month'] = np.cos(2 * np.pi * new_data['month'] / 12.0)

    return new_data

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
    model = load_model('models/best_model.keras')
    feature_scaler = joblib.load('models/feature_scaler.save')
    target_scaler = joblib.load('models/target_scaler.save')

    weather_data = fetch_weather_data()
    if weather_data is not None:
        new_data = process_weather_data(weather_data)

        # Update features list to use poa_global instead of global_irradiation
        features = ['temp_air', 'wind_speed', 'poa_global']

        # Use the predict_power function with all required arguments
        predicted_powers = predict_power(model, feature_scaler, target_scaler, new_data[features])

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
            
            # Set power to zero if global irradiation is zero
            if global_irradiation == 0:
                power[0] = 0
            
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
                'power_w': power[0] if new_data['poa_global'].iloc[i] != 0 else 0,
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
