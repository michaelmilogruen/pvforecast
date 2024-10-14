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
from tensorflow.keras.models import load_model
import requests
import json
from datetime import datetime

# Load the saved model and scalers
model = load_model('final_model.h5', compile=False)
sc_x = joblib.load('scaler_x.pkl')
sc_y = joblib.load('scaler_y.pkl')

def fetch_weather_data():
    """
    Fetch weather forecast data from the API.

    Returns:
        dict: The fetched weather forecast data as a dictionary, or None if the request failed.
    """
    lat_lon = "47.38770748541585,15.094127778561258"
    current_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    fc_start_time = current_time.strftime("%Y-%m-%dT%H:%M")
    fc_end_time = (current_time + pd.Timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M")

    url = f"https://dataset.api.hub.geosphere.at/v1/timeseries/forecast/nwp-v1-1h-2500m?lat_lon={lat_lon}&parameters=grad&parameters=t2m&parameters=sundur_acc&parameters=v10m&start={fc_start_time}&end={fc_end_time}"
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

    # Add a placeholder value for the first global_irradiation to match the length of other data
    global_irradiation.insert(0, 0)  # Adding 0 as a placeholder

    # Create DataFrame ensuring all arrays have the same length
    new_data = pd.DataFrame({
        'timestamp': pd.to_datetime(time_list),
        'temp_air': parameters['t2m']['data'],
        'wind_speed': np.abs(parameters['v10m']['data']),
        'global_irradiation': global_irradiation
    })

    # Extract hour and month features from the timestamp and apply circular encoding
    new_data['hour'] = new_data['timestamp'].dt.hour
    new_data['sin_hour'] = np.sin(2 * np.pi * new_data['hour'] / 24.0)
    new_data['cos_hour'] = np.cos(2 * np.pi * new_data['hour'] / 24.0)
    new_data['month'] = new_data['timestamp'].dt.month
    new_data['sin_month'] = np.sin(2 * np.pi * new_data['month'] / 12.0)
    new_data['cos_month'] = np.cos(2 * np.pi * new_data['month'] / 12.0)

    return new_data

def main():
    """
    Main function to fetch weather data, process it, and predict power output for the next day.
    """
    weather_data = fetch_weather_data()
    if weather_data is not None:
        new_data = process_weather_data(weather_data)

        # Define features to be used for prediction
        features = ['sin_hour', 'cos_hour', 'sin_month', 'cos_month', 'temp_air', 'wind_speed', 'global_irradiation']

        # Iterate over 24 hours and predict power for each hour separately
        predicted_powers = []
        for i in range(24):
            # Get features for each hour
            X_new = new_data[features].iloc[i].values.reshape(1, -1)

            # Scale the features using the loaded scaler
            X_new_scaled = sc_x.transform(X_new)

            # Reshape the data to fit the model's expected input shape (1, n_steps, n_features)
            X_new_scaled = X_new_scaled.reshape((1, 1, len(features)))

            # Make predictions using the loaded model
            prediction = model.predict(X_new_scaled)

            # Inverse transform the predicted value to get the actual power output
            predicted_power = sc_y.inverse_transform(prediction)
            predicted_powers.append(predicted_power[0][0])

        # Print the results to the console
        print("Predicted AC Power for the next day (in W):")
        print("| {:<20} | {:<15} | {:<15} | {:<15} | {:<25} |".format('Time', 'Power (W)', 'Temp (°C)', 'Wind Speed (m/s)', 'Global Irradiation (J/(m²*s))'))
        print("|" + "-" * 102 + "|")
        print("-" * 90)
        for i, power in enumerate(predicted_powers):
            future_time = (datetime.now() + pd.Timedelta(hours=i)).strftime("%Y-%m-%d %H:%M")
            temp_air = new_data['temp_air'].iloc[i]
            wind_speed = new_data['wind_speed'].iloc[i]
            global_irradiation = new_data['global_irradiation'].iloc[i]
            print("|  {:<18} |  {:<15.2f} |  {:<15.2f} |  {:<15.2f} |  {:<25.2f} |".format(future_time, power * 1000, temp_air, wind_speed, global_irradiation))

if __name__ == "__main__":
    main()