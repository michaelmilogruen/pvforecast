# -*- coding: utf-8 -*-
"""
Author: Michael Grün
Email: michaelgruen@hotmail.com
Description: This script loads a trained LSTM model and scalers to predict the power output of a photovoltaic system 
             using features downloaded from an API. Designed to be used both as a standalone script and as a module
             that can be imported by a UI application.
Version: 2.0
Date: 2025-03-21
"""

import numpy as np
import pandas as pd
import joblib
import requests
import json
import os
import csv
import glob
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pvlib
from pvlib.location import Location
from pvlib.clearsky import simplified_solis


class PVForecaster:
    """
    A class for forecasting PV power output using LSTM models.
    This class handles data fetching, preprocessing, and prediction in a way that's
    consistent with the training process in process_training_data.py.
    """
    
    def __init__(self, feature_set='station', sequence_length=24):
        """
        Initialize the PV Forecaster with model and configuration.
        
        Args:
            feature_set (str): Feature set to use ('inca', 'station', or 'combined')
            sequence_length (int): Sequence length for LSTM input
        """
        self.feature_set = feature_set
        self.sequence_length = sequence_length
        
        # Location parameters for Leoben
        self.latitude = 47.38770748541585
        self.longitude = 15.094127778561258
        self.altitude = 541
        self.tz = 'Etc/GMT+1'
        
        # Load model and scalers
        self._load_model_and_scalers()
        
        # Define feature sets
        self.feature_sets = self._define_feature_sets()
        
    def _load_model_and_scalers(self):
        """Load the trained model and necessary scalers from INFERENCE directory"""
        try:
            # Find the most recent model file
            model_files = glob.glob('models/INFERENCE/final_model_*.keras')
            if not model_files:
                model_files = glob.glob('models/INFERENCE/model_*.keras')
            
            if model_files:
                # Sort to get most recent (based on filename)
                model_files.sort(reverse=True)
                model_path = model_files[0]
                self.model = load_model(model_path)
                print(f"Loaded model from {model_path}")
                
                # Extract timestamp from model filename to find matching scalers
                if 'final_model_' in model_path:
                    timestamp = model_path.split('final_model_')[1].split('.keras')[0]
                else:
                    timestamp = model_path.split('model_')[1].split('.keras')[0]
                
                print(f"Using model timestamp: {timestamp}")
                
                # Load matching scalers
                minmax_path = f'models/INFERENCE/minmax_scaler_{timestamp}.pkl'
                standard_path = f'models/INFERENCE/standard_scaler_{timestamp}.pkl'
                target_path = f'models/INFERENCE/target_scaler_{timestamp}.pkl'
                robust_path = f'models/INFERENCE/robust_scaler_{timestamp}.pkl'
                
                # Load scalers if files exist
                if os.path.exists(minmax_path):
                    self.minmax_scaler = joblib.load(minmax_path)
                    print(f"Loaded MinMax scaler from: {minmax_path}")
                else:
                    # Try to find any minmax scaler
                    minmax_files = glob.glob('models/INFERENCE/minmax_scaler_*.pkl')
                    if minmax_files:
                        minmax_files.sort(reverse=True)
                        self.minmax_scaler = joblib.load(minmax_files[0])
                        print(f"Loaded MinMax scaler from: {minmax_files[0]}")
                    else:
                        self.minmax_scaler = joblib.load('models/minmax_scaler.pkl')
                        print("Loaded MinMax scaler from default path")
                
                if os.path.exists(standard_path):
                    self.standard_scaler = joblib.load(standard_path)
                    print(f"Loaded Standard scaler from: {standard_path}")
                else:
                    # Try to find any standard scaler
                    standard_files = glob.glob('models/INFERENCE/standard_scaler_*.pkl')
                    if standard_files:
                        standard_files.sort(reverse=True)
                        self.standard_scaler = joblib.load(standard_files[0])
                        print(f"Loaded Standard scaler from: {standard_files[0]}")
                    else:
                        self.standard_scaler = joblib.load('models/standard_scaler.pkl')
                        print("Loaded Standard scaler from default path")
                
                if os.path.exists(target_path):
                    self.target_scaler = joblib.load(target_path)
                    print(f"Loaded Target scaler from: {target_path}")
                else:
                    # Try to find any target scaler
                    target_files = glob.glob('models/INFERENCE/target_scaler_*.pkl')
                    if target_files:
                        target_files.sort(reverse=True)
                        self.target_scaler = joblib.load(target_files[0])
                        print(f"Loaded Target scaler from: {target_files[0]}")
                    else:
                        self.target_scaler = joblib.load('models/target_scaler.pkl')
                        print("Loaded Target scaler from default path")
                
                # Optionally load robust scaler if it exists
                if os.path.exists(robust_path):
                    self.robust_scaler = joblib.load(robust_path)
                    print(f"Loaded Robust scaler from: {robust_path}")
                else:
                    # Try to find any robust scaler
                    robust_files = glob.glob('models/INFERENCE/robust_scaler_*.pkl')
                    if robust_files:
                        robust_files.sort(reverse=True)
                        self.robust_scaler = joblib.load(robust_files[0])
                        print(f"Loaded Robust scaler from: {robust_files[0]}")
            else:
                # Fall back to default paths if no model files found in INFERENCE directory
                print("No model files found in INFERENCE directory, falling back to defaults.")
                self.model = load_model('models/power_forecast_model.keras')
                print("Loaded model from default path")
                
                # Load default scalers
                self.minmax_scaler = joblib.load('models/minmax_scaler.pkl')
                self.standard_scaler = joblib.load('models/standard_scaler.pkl')
                self.target_scaler = joblib.load('models/target_scaler.pkl')
                print("Loaded scalers from default paths")
            
            print("Model and scalers loaded successfully.")
        except Exception as e:
            print(f"Error loading model or scalers: {e}")
            raise
    
    def _define_feature_sets(self):
        """Define the feature sets used in training"""
        return {
            'inca': [
                'INCA_GlobalRadiation [W m-2]',
                'INCA_Temperature [degree_Celsius]',
                'INCA_WindSpeed [m s-1]',
                'INCA_ClearSkyIndex',
                'hour_sin',
                'hour_cos',
                'day_cos',
                'day_sin',
                'isNight'
            ],
            'station': [
                'GlobalRadiation [W m-2]',
                'Temperature [degree_Celsius]',
                'WindSpeed [m s-1]',
                'ClearSkyIndex',
                'hour_sin',
                'hour_cos',
                'day_cos',
                'day_sin',
                'isNight'
            ],
            'combined': [
                'Combined_GlobalRadiation [W m-2]',
                'Combined_Temperature [degree_Celsius]',
                'Combined_WindSpeed [m s-1]',
                'Combined_ClearSkyIndex',
                'hour_sin',
                'hour_cos',
                'day_cos',
                'day_sin',
                'isNight'
            ]
        }
    
    def fetch_weather_data(self, hours=60):
        """
        Fetch weather forecast data from the API.
        
        Args:
            hours (int): Number of hours to forecast
            
        Returns:
            dict: The fetched weather forecast data as a dictionary, or None if the request failed.
        """
        lat_lon = f"{self.latitude},{self.longitude}"
        current_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        fc_start_time = current_time.strftime("%Y-%m-%dT%H:%M")
        fc_end_time = (current_time + pd.Timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M")
        
        url = f"https://dataset.api.hub.geosphere.at/v1/timeseries/forecast/nwp-v1-1h-2500m?lat_lon={lat_lon}&parameters=cape&parameters=cin&parameters=grad&parameters=mnt2m&parameters=mxt2m&parameters=rain_acc&parameters=rh2m&parameters=rr_acc&parameters=snow_acc&parameters=snowlmt&parameters=sp&parameters=sundur_acc&parameters=sy&parameters=t2m&parameters=tcc&parameters=u10m&parameters=ugust&parameters=v10m&parameters=vgust&start={fc_start_time}&end={fc_end_time}"
        
        try:
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                print("Weather data fetched successfully.")
                return data
            else:
                print(f"Failed to fetch data. Status code: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return None
    
    def process_weather_data(self, data):
        """
        Process the fetched weather forecast data and create a DataFrame.
        
        Args:
            data (dict): The fetched weather forecast data.
            
        Returns:
            pandas.DataFrame: A DataFrame containing the processed weather data with extracted features.
        """
        time_list = data['timestamps']
        parameters = data['features'][0]['properties']['parameters']
        
        # Calculate global irradiation difference and scale to W/m²
        global_irradiation = [(parameters['grad']['data'][i + 1] - val) / 3600 for i, val in enumerate(parameters['grad']['data'][:-1])]
        global_irradiation.insert(0, 0)  # Adding 0 as a placeholder
        
        # Create DataFrame with all parameters
        new_data = pd.DataFrame({
            'timestamp': pd.to_datetime(time_list),
            'temp_air': parameters['t2m']['data'],
            'wind_speed': np.sqrt(np.array(parameters['u10m']['data'])**2 + np.array(parameters['v10m']['data'])**2),  # Proper wind speed calculation
            'poa_global': global_irradiation,
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
        
        # Create essential column mappings immediately to prevent KeyErrors later
        if 'GlobalRadiation [W m-2]' not in new_data.columns:
            new_data['GlobalRadiation [W m-2]'] = new_data['poa_global']
            print("Created GlobalRadiation [W m-2] column from poa_global in process_weather_data")
            
        if 'Temperature [degree_Celsius]' not in new_data.columns:
            new_data['Temperature [degree_Celsius]'] = new_data['temp_air']
            
        if 'WindSpeed [m s-1]' not in new_data.columns:
            new_data['WindSpeed [m s-1]'] = new_data['wind_speed']
        
        return new_data
    
    def add_derived_features(self, df):
        """
        Add derived time-based features consistent with the training process.
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            DataFrame with added time-based features
        """
        # Ensure all required columns exist before calculations begin
        print("Adding essential feature mappings before derived calculations")
        if 'GlobalRadiation [W m-2]' not in df.columns and 'poa_global' in df.columns:
            df['GlobalRadiation [W m-2]'] = df['poa_global']
            print("Created GlobalRadiation [W m-2] from poa_global")
            
        if 'INCA_GlobalRadiation [W m-2]' not in df.columns and 'poa_global' in df.columns:
            df['INCA_GlobalRadiation [W m-2]'] = df['poa_global']
            
        if 'Combined_GlobalRadiation [W m-2]' not in df.columns and 'poa_global' in df.columns:
            df['Combined_GlobalRadiation [W m-2]'] = df['poa_global']
            
        # Print available columns for debugging
        print("Columns before derived calculations:", df.columns.tolist())
            
        # Calculate hour with minute resolution (e.g., 15:15 becomes 15.25)
        df['hour'] = df.index.hour + df.index.minute / 60
        df['day_of_year'] = df.index.dayofyear
        
        # Add circular encoding for hour and day of year
        angle_hour = 2 * np.pi * df['hour'] / 24
        df['hour_sin'] = np.sin(angle_hour)
        df['hour_cos'] = np.cos(angle_hour)
        
        df['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
        df['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)
        
        # Add isNight feature based on solar position
        df = self.calculate_clear_sky(df)
        
        return df
    
    def calculate_clear_sky(self, df):
            """
            Calculate ClearSkyIndex and append it to the given dataframe.
            
            Args:
                df: DataFrame with datetime index
                
            Returns:
                DataFrame with added ClearSkyIndex
            """
            # Define location
            location = Location(self.latitude, self.longitude, self.tz, self.altitude)
            
            # Get solar position
            solar_position = location.get_solarposition(df.index)
            
            # Calculate pressure based on altitude and convert from hPa to Pa
            pressure = pvlib.atmosphere.alt2pres(self.altitude) * 100
            
            # Approximate precipitable water (in cm)
            precipitable_water = 1.0  # Default value as a simple approximation
            
            # Calculate apparent elevation and ensure it's not negative
            apparent_elevation = 90 - solar_position['apparent_zenith']
            apparent_elevation = apparent_elevation.clip(lower=0)
            
            # Calculate clear sky irradiance using simplified Solis model
            clear_sky = simplified_solis(
                apparent_elevation=apparent_elevation,
                aod700=0.1,  # Default value for aerosol optical depth
                precipitable_water=precipitable_water,
                pressure=pressure
            )
            
            # Set irradiance components to 0 for nighttime (elevation <= 0)
            night_mask = (apparent_elevation <= 0)
            clear_sky.loc[night_mask, 'ghi'] = 0
            
            # Set other irradiance components to 0 for nighttime if they exist
            if 'dni' in clear_sky.columns:
                clear_sky.loc[night_mask, 'dni'] = 0
            if 'dhi' in clear_sky.columns:
                clear_sky.loc[night_mask, 'dhi'] = 0
            
            # Add isNight feature (1 for night, 0 for day)
            df['isNight'] = night_mask.astype(int)
            
            # Calculate ClearSkyIndex based on feature set
            if self.feature_set == 'inca':
                radiation_col = 'INCA_GlobalRadiation [W m-2]'
                df['INCA_ClearSkyIndex'] = np.where(
                    clear_sky['ghi'] > 10,
                    df[radiation_col] / clear_sky['ghi'],
                    0
                )
                df['INCA_ClearSkyIndex'] = df['INCA_ClearSkyIndex'].clip(0, 1.5)
                
            elif self.feature_set == 'station':
                radiation_col = 'GlobalRadiation [W m-2]'
                df['ClearSkyIndex'] = np.where(
                    clear_sky['ghi'] > 10,
                    df[radiation_col] / clear_sky['ghi'],
                    0
                )
                df['ClearSkyIndex'] = df['ClearSkyIndex'].clip(0, 1.5)
                
            else:  # combined
                # INCA
                inca_rad_col = 'INCA_GlobalRadiation [W m-2]'
                df['INCA_ClearSkyIndex'] = np.where(
                    clear_sky['ghi'] > 10,
                    df[inca_rad_col] / clear_sky['ghi'],
                    0
                )
                df['INCA_ClearSkyIndex'] = df['INCA_ClearSkyIndex'].clip(0, 1.5)
                
                # Station
                rad_col = 'GlobalRadiation [W m-2]'
                df['ClearSkyIndex'] = np.where(
                    clear_sky['ghi'] > 10,
                    df[rad_col] / clear_sky['ghi'],
                    0
                )
                df['ClearSkyIndex'] = df['ClearSkyIndex'].clip(0, 1.5)
                
                # Combined
                df['Combined_ClearSkyIndex'] = df[['INCA_ClearSkyIndex', 'ClearSkyIndex']].mean(axis=1)
                
            return df
    
    def normalize_data(self, df):
        """
        Normalize the data using the same approach as in training.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with normalized features
        """
        # Define which columns should be normalized with which scaler
        # This should match the approach in process_training_data.py
        time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'isNight']
        
        # Check which features need MinMaxScaler vs StandardScaler
        minmax_columns = [
            # INCA Radiation features
            'INCA_GlobalRadiation [W m-2]',
            'INCA_ClearSkyGHI',
            'INCA_ClearSkyDHI',  # Added missing feature
            'INCA_ClearSkyDNI',  # Added missing feature
            # Station Radiation features
            'GlobalRadiation [W m-2]',
            'ClearSkyGHI',
            # Combined Radiation features
            'Combined_GlobalRadiation [W m-2]',
        ]
        
        standard_columns = [
            # INCA weather measurements
            'INCA_Temperature [degree_Celsius]',
            'INCA_WindSpeed [m s-1]',
            'INCA_ClearSkyIndex',
            # Station weather measurements
            'Temperature [degree_Celsius]',
            'WindSpeed [m s-1]',
            'ClearSkyIndex',
            # Combined measurements
            'Combined_Temperature [degree_Celsius]',
            'Combined_WindSpeed [m s-1]',
            'Combined_ClearSkyIndex'
        ]
        
        # Filter columns that actually exist in the dataframe
        minmax_columns = [col for col in minmax_columns if col in df.columns]
        standard_columns = [col for col in standard_columns if col in df.columns]
        
        # Apply scalers
        if minmax_columns:
            df[minmax_columns] = self.minmax_scaler.transform(df[minmax_columns])
        
        if standard_columns:
            df[standard_columns] = self.standard_scaler.transform(df[standard_columns])
        
        return df
    def prepare_data_for_prediction(self, weather_data=None, hours=60):
        """
        Prepare data for prediction by fetching weather data if not provided,
        processing it, and normalizing it.
        
        Args:
            weather_data (dict, optional): Weather data from API. If None, it will be fetched.
            hours (int): Number of hours to forecast if weather_data is None
            
        Returns:
            tuple: (processed_df, features_array) where processed_df is the full processed DataFrame
                  and features_array is the normalized features ready for the model
        """
        # Fetch weather data if not provided
        if weather_data is None:
            weather_data = self.fetch_weather_data(hours=hours)
            if weather_data is None:
                raise ValueError("Failed to fetch weather data")
        
        # Process the weather data
        df = self.process_weather_data(weather_data)
        
        # Verify poa_global exists in the processed data
        if 'poa_global' not in df.columns:
            raise KeyError("'poa_global' column is missing in the processed weather data. Check the API response structure.")
        
        # Print columns for debugging
        print("Data columns after processing:", df.columns.tolist())
        
        # Add derived features
        df = self.add_derived_features(df)
        
        # Proactively add all required features to ensure they're available
        print("Adding all required features to ensure compatibility with the model")
        
        # Add all required features from all feature sets
        # Common time features are already added in add_derived_features
        
        # Station features
        try:
            if 'GlobalRadiation [W m-2]' not in df.columns:
                df['GlobalRadiation [W m-2]'] = df['poa_global']
            if 'Temperature [degree_Celsius]' not in df.columns:
                df['Temperature [degree_Celsius]'] = df['temp_air']
            if 'WindSpeed [m s-1]' not in df.columns:
                df['WindSpeed [m s-1]'] = df['wind_speed']
            if 'ClearSkyIndex' not in df.columns:
                df['ClearSkyIndex'] = df['total_cloud_cover']
        except KeyError as e:
            print(f"Error creating station features: {e}")
            print(f"Available columns: {df.columns.tolist()}")
            raise KeyError(f"Cannot create necessary feature mappings. Missing column: {e}")
        
        # INCA features
        try:
            if 'INCA_GlobalRadiation [W m-2]' not in df.columns:
                df['INCA_GlobalRadiation [W m-2]'] = df['poa_global']
            if 'INCA_Temperature [degree_Celsius]' not in df.columns:
                df['INCA_Temperature [degree_Celsius]'] = df['temp_air']
            if 'INCA_WindSpeed [m s-1]' not in df.columns:
                df['INCA_WindSpeed [m s-1]'] = df['wind_speed']
            if 'INCA_ClearSkyIndex' not in df.columns:
                df['INCA_ClearSkyIndex'] = df['total_cloud_cover']
            if 'INCA_ClearSkyDHI' not in df.columns:
                df['INCA_ClearSkyDHI'] = 0.0
            if 'INCA_ClearSkyDNI' not in df.columns:
                df['INCA_ClearSkyDNI'] = 0.0
            if 'INCA_ClearSkyGHI' not in df.columns:
                df['INCA_ClearSkyGHI'] = 0.0
        except KeyError as e:
            print(f"Error creating INCA features: {e}")
            print(f"Available columns: {df.columns.tolist()}")
            raise KeyError(f"Cannot create necessary INCA feature mappings. Missing column: {e}")
        
        # Combined features
        try:
            if 'Combined_GlobalRadiation [W m-2]' not in df.columns:
                df['Combined_GlobalRadiation [W m-2]'] = df['poa_global']
            if 'Combined_Temperature [degree_Celsius]' not in df.columns:
                df['Combined_Temperature [degree_Celsius]'] = df['temp_air']
            if 'Combined_WindSpeed [m s-1]' not in df.columns:
                df['Combined_WindSpeed [m s-1]'] = df['wind_speed']
            if 'Combined_ClearSkyIndex' not in df.columns:
                df['Combined_ClearSkyIndex'] = df['total_cloud_cover']
        except KeyError as e:
            print(f"Error creating Combined features: {e}")
            print(f"Available columns: {df.columns.tolist()}")
            raise KeyError(f"Cannot create necessary Combined feature mappings. Missing column: {e}")
        
        # Normalize the data
        df = self.normalize_data(df)
        
        # Get the features for the selected feature set
        features = self.feature_sets[self.feature_set]
        
        # Extract features as numpy array
        X = df[features].values
        
        return df, X
    
    def create_sequences(self, X, seq_length):
        """
        Create sequences for LSTM input.
        
        Args:
            X: Feature array
            seq_length: Sequence length
            
        Returns:
            numpy.ndarray: Sequences for LSTM input
        """
        sequences = []
        for i in range(len(X) - seq_length + 1):
            sequences.append(X[i:(i + seq_length)])
        
        return np.array(sequences)
    
    def predict(self, weather_data=None, hours=60, return_df=True):
        """
        Predict PV power output.
        
        Args:
            weather_data (dict, optional): Weather data from API. If None, it will be fetched.
            hours (int): Number of hours to forecast if weather_data is None
            return_df (bool): If True, return a DataFrame with predictions and features
            
        Returns:
            If return_df is True:
                pandas.DataFrame: DataFrame with predictions and features
            Else:
                numpy.ndarray: Predicted power values
        """
        try:
            # Prepare data for prediction
            df, X = self.prepare_data_for_prediction(weather_data, hours)
            
            # Print available columns for debugging
            print("Available columns for prediction:", df.columns.tolist())
            
            # Ensure required columns exist
            required_columns = ['poa_global', 'temp_air', 'wind_speed', 'total_cloud_cover',
                               'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'isNight']
            
            for col in required_columns:
                if col not in df.columns:
                    raise KeyError(f"Required column '{col}' not found in processed data. Check the API response structure.")
            
            # Create a new DataFrame with the same feature names as used during training
            # Map the inference feature names to the training feature names
            training_data = pd.DataFrame({
                'GlobalRadiation [W m-2]': df['poa_global'],
                'Temperature [degree_Celsius]': df['temp_air'],
                'WindSpeed [m s-1]': df['wind_speed'],
                'ClearSkyIndex': df['total_cloud_cover'],  # Use cloud cover as ClearSkyIndex
                'hour_sin': df['hour_sin'],
                'hour_cos': df['hour_cos'],
                'day_sin': df['day_sin'],
                'day_cos': df['day_cos'],
                'isNight': df['isNight']
            })
            
            # Define the features in the same order as used during training
            training_features = [
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
            
            # Use the training data for prediction
            X = training_data[training_features].values
            
            # Create sequences
            X_seq = self.create_sequences(X, self.sequence_length)
            
            # Make predictions
            y_pred = self.model.predict(X_seq)
            
            # Inverse transform predictions
            y_pred_inv = self.target_scaler.inverse_transform(y_pred)
            
            # Ensure non-negative power values
            y_pred_inv = np.maximum(y_pred_inv, 0)
        except Exception as e:
            print(f"Error running forecast: {e}")
            raise
        
        if return_df:
            # Create a DataFrame with predictions and features
            # The predictions start after sequence_length-1 timesteps
            pred_start_idx = self.sequence_length - 1
            pred_df = df.iloc[pred_start_idx:pred_start_idx + len(y_pred)].copy()
            pred_df['power_w'] = y_pred_inv
            
            # Reset index to have timestamp as a column
            pred_df.reset_index(inplace=True)
            
            return pred_df
        else:
            return y_pred_inv
    
    def plot_forecast(self, pred_df, save_path=None, show=True):
        """
        Plot the forecasted power output.
        
        Args:
            pred_df: DataFrame with predictions
            save_path (str, optional): Path to save the plot
            show (bool): Whether to show the plot
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot power output
        ax1.plot(pred_df['timestamp'], pred_df['power_w'], 'b-', label='Predicted Power (W)')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Power (W)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        # Create a second y-axis for irradiance
        ax2 = ax1.twinx()
        if self.feature_set == 'inca':
            irradiance_col = 'INCA_GlobalRadiation [W m-2]'
        elif self.feature_set == 'station':
            irradiance_col = 'GlobalRadiation [W m-2]'
        else:
            irradiance_col = 'Combined_GlobalRadiation [W m-2]'
            
        # Get the original (unnormalized) irradiance values
        # We need to inverse transform the normalized values
        # Create a simple one-column array for the irradiance values
        irradiance_values = pred_df[irradiance_col].values.reshape(-1, 1)
        
        # Create a separate scaler just for this column to avoid dimension issues
        from sklearn.preprocessing import MinMaxScaler
        irradiance_scaler = MinMaxScaler()
        irradiance_scaler.min_ = np.array([0])  # Assuming normalized values are between 0 and 1
        irradiance_scaler.scale_ = np.array([1])
        
        # Use the direct values without inverse transformation for now
        # This is a simplification - in a real application, you might want to
        # properly inverse transform using the original scaler parameters
        
        ax2.plot(pred_df['timestamp'], pred_df[irradiance_col], 'r-', label='Global Irradiance (W/m²)')
        ax2.set_ylabel('Irradiance (W/m²)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Add night periods as shaded areas
        night_periods = []
        current_night = None
        for i, row in pred_df.iterrows():
            if row['isNight'] == 1 and current_night is None:
                current_night = i
            elif row['isNight'] == 0 and current_night is not None:
                night_periods.append((current_night, i))
                current_night = None
        
        # Add the last night period if it extends to the end
        if current_night is not None:
            night_periods.append((current_night, len(pred_df) - 1))
        
        for start, end in night_periods:
            ax1.axvspan(pred_df['timestamp'].iloc[start], pred_df['timestamp'].iloc[end], 
                        alpha=0.2, color='gray')
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.title(f'PV Power Forecast ({self.feature_set.capitalize()})')
        plt.grid(True, alpha=0.3)
        
        # Format x-axis to show dates nicely
        fig.autofmt_xdate()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def export_to_csv(self, pred_df, filename='data/forecast_data.csv'):
        """
        Export predictions to CSV.
        
        Args:
            pred_df: DataFrame with predictions
            filename: Path to save the CSV file
            
        Returns:
            str: Path to the saved file
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save to CSV
        pred_df.to_csv(filename, index=False)
        print(f"Forecast data exported to {filename}")
        
        return filename
    
    def export_to_json(self, pred_df, filename='data/forecast_data.json'):
        """
        Export predictions to JSON.
        
        Args:
            pred_df: DataFrame with predictions
            filename: Path to save the JSON file
            
        Returns:
            str: Path to the saved file
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Convert datetime to string
        pred_df_copy = pred_df.copy()
        pred_df_copy['timestamp'] = pred_df_copy['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Save to JSON
        pred_df_copy.to_json(filename, orient='records', date_format='iso')
        print(f"Forecast data exported to {filename}")
        
        return filename


def main():
    """
    Main function to run the forecaster as a standalone script.
    """
    # Create forecaster with default settings
    forecaster = PVForecaster(feature_set='station', sequence_length=24)
    
    # Make predictions
    try:
        pred_df = forecaster.predict(hours=60)
        
        # Plot forecast
        forecaster.plot_forecast(pred_df, save_path='results/forecast_plot.png')
        
        # Export to CSV
        forecaster.export_to_csv(pred_df)
        
        # Export to JSON
        forecaster.export_to_json(pred_df)
        
        # Print summary
        print("\nForecast Summary:")
        print(f"Feature set: {forecaster.feature_set}")
        print(f"Sequence length: {forecaster.sequence_length}")
        print(f"Forecast period: {pred_df['timestamp'].min()} to {pred_df['timestamp'].max()}")
        print(f"Maximum predicted power: {pred_df['power_w'].max():.2f} W")
        print(f"Average predicted power: {pred_df['power_w'].mean():.2f} W")
        print(f"Total predicted energy: {pred_df['power_w'].sum() / 1000:.2f} kWh")
        
    except Exception as e:
        print(f"Error making predictions: {e}")


if __name__ == "__main__":
    main()
