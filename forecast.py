# -*- coding: utf-8 -*-
"""
Author: Michael Grün
Email: michaelgruen@hotmail.com
Description: This script fetches weather forecast data from an API and processes it to create an Excel file
             with the forecast data for a specified number of hours.
Version: 1.0
Date: 2024-05-04
"""

import json
import pandas as pd
import requests
from datetime import datetime, timedelta

def get_user_input(hours: int) -> tuple:
    """
    Get user input for forecasting start and end times.

    Args:
        hours (int): The number of hours to forecast.

    Returns:
        tuple: A tuple containing the forecasting start time and end time.
    """
    current_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    
    fc_start_time = current_time.strftime("%Y-%m-%dT%H:%M")
    fc_end_time = (current_time + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M")
    
    return fc_start_time, fc_end_time


def fetch_weather_data(lat_lon, fc_start_time, fc_end_time):
    """
    Fetch weather forecast data from the API.

    Args:
        lat_lon (str): Latitude and longitude for the location.
        fc_start_time (str): Forecasting start time in the format "YYYY-MM-DDThh:mm".
        fc_end_time (str): Forecasting end time in the format "YYYY-MM-DDThh:mm".

    Returns:
        dict: The fetched weather forecast data as a dictionary.
    """
    url = f"https://dataset.api.hub.geosphere.at/v1/timeseries/forecast/nwp-v1-1h-2500m?lat_lon={lat_lon}&parameters=grad&parameters=t2m&parameters=sundur_acc&parameters=v10m&start={fc_start_time}&end={fc_end_time}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        file_path = "weather_forecast.json"
        with open(file_path, "w") as json_file:
            json_file.write(response.text)
        print("JSON data saved successfully.")
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
        pandas.DataFrame: A DataFrame containing the processed weather data.
    """
    with open('weather_forecast.json') as f:
        data = json.load(f)

    df = pd.json_normalize(data['features'], meta=['properties', ['parameters']])
    time_list = data['timestamps']

    df1 = df[['properties.parameters.grad.data',
              'properties.parameters.t2m.data',
              'properties.parameters.sundur_acc.data',
              'properties.parameters.v10m.data']]

    list1 = list(df1['properties.parameters.grad.data'][0])
    list2 = list(df1['properties.parameters.t2m.data'][0])
    list3 = list(df1['properties.parameters.sundur_acc.data'][0])
    list4 = list(df1['properties.parameters.v10m.data'][0])

    df2 = pd.DataFrame()
    df2['timestamp'] = time_list
    df2['surface global radiation [J/m²]'] = list1
    df2['Temperture 2m above ground [°C]'] = list2
    df2['sunshine duration accumulated [s]'] = list3
    df2['wind speed northern direction [m/s]'] = list4

    return df2


def save_data_to_excel(df):
    """
    Save the processed weather data to an Excel file.

    Args:
        df (pandas.DataFrame): The DataFrame containing the processed weather data.
    """
    with pd.ExcelWriter("forecast_data.xlsx") as writer:
        df.to_excel(writer, sheet_name='Forecast_Data1')
    print("Data saved to Excel file successfully.")


def main():
    """
    Main function to run the script.
    """
    f_ref_time = 1
    lat_lon = "47.38770748541585,15.094127778561258"



    fc_start_time, fc_end_time = get_user_input(24)

    weather_data = fetch_weather_data(lat_lon, fc_start_time, fc_end_time)
    if weather_data is not None:
        df = process_weather_data(weather_data)
        print(df)
        save_data_to_excel(df)


if __name__ == "__main__":
    main()

