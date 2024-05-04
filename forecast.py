# -*- coding: utf-8 -*-
"""
Author: Michael Grün
Email: michaelgruen@hotmail.com
Description: This script calculates the power output of a photovoltaic system
             using the PVLib library and processes the results in an Excel file.
Version: 1.0
Date: 2024-05-04
"""

import json
import pandas as pd

f_ref_time = 1
# -------------- format of fc_start_time and fc_end_time:  (YYYY-MM-DDThh:mm) -------------------- 
fc_start_time = "2024-04-07T06:00"
fc_end_time = "2024-04-09T06:00"

#fc_start_time = float(input("Please input forecasting start time (YYYY-MM-DDThh:mm): ")) #e.g.: 2024-04-05T10:00
#fc_end_time = float(input("Please input forecasting end time (YYYY-MM-DDThh:mm): ")) #e.g.: 2024-04-06T20:00

# Latitude and lonfitude for Leoben EVT
lat_lon = "47.38770748541585,15.094127778561258"

import requests

# URL of the JSON data
#url = "https://dataset.api.hub.geosphere.at/v1/timeseries/forecast/nwp-v1-1h-2500m?lat_lon=47.38770748541585,15.094127778561258&parameters=grad&parameters=t2m&parameters=sundur_acc&parameters=v10m&start=2024-04-05T10:00&end=2024-04-06T20:00"
url = "https://dataset.api.hub.geosphere.at/v1/timeseries/forecast/nwp-v1-1h-2500m?lat_lon="+lat_lon+"&parameters=grad&parameters=t2m&parameters=sundur_acc&parameters=v10m&start="+fc_start_time+"&end="+fc_end_time


# Send request and get JSON data
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Get the JSON data
    data = response.json()

    # Specify the file path to save the JSON data
    file_path = "weather_forecast.json"

    # Write the JSON data to a file
    with open(file_path, "w") as json_file:
        json_file.write(response.text)

    print("JSON data saved successfully.")
else:
    print(f"Failed to fetch data. Status code: {response.status_code}")
    
    

with open('weather_forecast.json') as f:
    data = json.load(f)

# Use pd.json_normalize to convert the JSON to a DataFrame
# df = pd.json_normalize(data['features'],
#                     meta=['properties',['parameters',['grad','name'],['grad','unit'],['grad','data'],['t2m','name'], ['t2m','unit'], ['t2m', 'data']]])

df = pd.json_normalize(data['features'],
                    meta=['properties',['parameters']])
                    
time_list = data['timestamps']

# Remodeling of the data to a dataframe 
df1 = df[['properties.parameters.grad.data', 
          'properties.parameters.t2m.data',
          'properties.parameters.sundur_acc.data',
          'properties.parameters.v10m.data']]


list1 = list(df1['properties.parameters.grad.data'][0]) #make a list with the stored data in the first field with the index 'properties.parameters.grad.data'
list2 = list(df1['properties.parameters.t2m.data'][0])
list3 = list(df1['properties.parameters.sundur_acc.data'][0])
list4 = list(df1['properties.parameters.v10m.data'][0])

# list1_neu = list1[0]
# list2_neu = list2[0]
# list3_neu = list3[0]
# list4_neu = list4[0]

df2 = pd.DataFrame()
df2['timestamp'] = time_list
df2['surface global radiation [J/m²]'] = list1
df2['Temperture 2m above ground [°C]'] = list2
df2['sunshine duration accumulated [s]'] = list3
df2['wind speed northern direction [m/s]'] = list4

# Display the DataFrame df2
print(df2)

#Export data from df to Excel-file
with pd.ExcelWriter("forecast_data.xlsx") as writer: 
    df2.to_excel(writer, sheet_name='Forecast_Data1')