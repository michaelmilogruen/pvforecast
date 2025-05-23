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
    
    def __init__(self, sequence_length=24):
        """
        Initialize the PV Forecaster with model and configuration.
        
        Args:
            sequence_length (int): Sequence length for LSTM input
        """
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
        """Define the features used in training"""
        # Single standard feature set for all prediction scenarios
        return [
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
    
    def fetch_weather_data(self, hours=24):
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
        fc_end_time = (current_time + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M")
        
        print(f"Requesting forecast from {fc_start_time} to {fc_end_time} ({hours} hours)")
        
        url = f"https://dataset.api.hub.geosphere.at/v1/timeseries/forecast/nwp-v1-1h-2500m?lat_lon={lat_lon}&parameters=cape&parameters=cin&parameters=grad&parameters=mnt2m&parameters=mxt2m&parameters=rain_acc&parameters=rh2m&parameters=rr_acc&parameters=snow_acc&parameters=snowlmt&parameters=sp&parameters=sundur_acc&parameters=sy&parameters=t2m&parameters=tcc&parameters=u10m&parameters=ugust&parameters=v10m&parameters=vgust&start={fc_start_time}&end={fc_end_time}"
        
        try:
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                print(f"Weather data fetched successfully for {hours} hours.")
                print(f"Received {len(data['timestamps'])} data points.")
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
        # Ensure required columns exist before calculations begin
        print("Adding essential feature mappings before derived calculations")
        if 'GlobalRadiation [W m-2]' not in df.columns and 'poa_global' in df.columns:
            df['GlobalRadiation [W m-2]'] = df['poa_global']
            print("Created GlobalRadiation [W m-2] from poa_global")
            
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
        
        # Calculate ClearSkyIndex from total_cloud_cover and add isNight feature
        df = self.calculate_clear_sky(df)
        
        return df
    
    def calculate_clear_sky(self, df):
            """
            Calculate ClearSkyIndex directly from total_cloud_cover and add isNight feature.
            
            Args:
                df: DataFrame with datetime index
                
            Returns:
                DataFrame with added ClearSkyIndex and isNight features
            """
            # Define location
            location = Location(self.latitude, self.longitude, self.tz, self.altitude)
            
            # Get solar position
            solar_position = location.get_solarposition(df.index)
            
            # Calculate apparent elevation and ensure it's not negative
            apparent_elevation = 90 - solar_position['apparent_zenith']
            apparent_elevation = apparent_elevation.clip(lower=0)
            
            # Add isNight feature (1 for night, 0 for day)
            night_mask = (apparent_elevation <= 0)
            df['isNight'] = night_mask.astype(int)
            
            # Ensure we have GlobalRadiation [W m-2]
            if 'GlobalRadiation [W m-2]' not in df.columns and 'poa_global' in df.columns:
                df['GlobalRadiation [W m-2]'] = df['poa_global']
                print("Created GlobalRadiation [W m-2] from poa_global in calculate_clear_sky")
            
            # Calculate ClearSkyIndex directly from total_cloud_cover
            if 'total_cloud_cover' in df.columns:
                # Use 1.0 - total_cloud_cover as ClearSkyIndex
                # This creates an inverse relationship where:
                # - 0 cloud cover = 1.0 clear sky index (perfectly clear)
                # - 1 cloud cover = 0.0 clear sky index (completely cloudy)
                df['ClearSkyIndex'] = 1.0 - df['total_cloud_cover']
                print("Created ClearSkyIndex directly from total_cloud_cover")
            else:
                print("Warning: total_cloud_cover not found. Setting ClearSkyIndex to default value.")
                df['ClearSkyIndex'] = 0.5  # Set a default middle value if no cloud cover data
            
            return df
    
    def normalize_data(self, df):
        """
        Normalize the data using the same approach as in training.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with normalized features
        """
        # Check for required features at the start
        if 'ClearSkyIndex' not in df.columns:
            print("WARNING: ClearSkyIndex not found at start of normalization")
        
        # Define the standard feature columns used in training
        minmax_columns = [
            'GlobalRadiation [W m-2]'
        ]
        
        standard_columns = [
            'Temperature [degree_Celsius]',
            'WindSpeed [m s-1]',
            'ClearSkyIndex'
        ]
        
        # Time features are typically not normalized
        time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'isNight']
        
        # Ensure all required columns exist
        missing_columns = []
        for col in minmax_columns + standard_columns + time_features:
            if col not in df.columns:
                missing_columns.append(col)
                
        if missing_columns:
            print(f"Warning: Missing required columns for normalization: {missing_columns}")
            print(f"Available columns: {df.columns.tolist()}")
            # Create missing columns with defaults if possible
            for col in missing_columns:
                if col == 'ClearSkyIndex' and 'total_cloud_cover' in df.columns:
                    df[col] = 1.0 - df['total_cloud_cover']
                    print(f"Created {col} from total_cloud_cover")
                    missing_columns.remove(col)
                elif col == 'GlobalRadiation [W m-2]' and 'poa_global' in df.columns:
                    df[col] = df['poa_global']
                    print(f"Created {col} from poa_global")
                    missing_columns.remove(col)
                elif col == 'Temperature [degree_Celsius]' and 'temp_air' in df.columns:
                    df[col] = df['temp_air']
                    print(f"Created {col} from temp_air")
                    missing_columns.remove(col)
                elif col == 'WindSpeed [m s-1]' and 'wind_speed' in df.columns:
                    df[col] = df['wind_speed']
                    print(f"Created {col} from wind_speed")
                    missing_columns.remove(col)
            
            # If there are still missing columns, raise an error
            if missing_columns:
                raise KeyError(f"Cannot normalize data. Missing required columns: {missing_columns}")
            
            # Debug: Check columns after creating missing ones
            print("Columns after creating missing ones:", df.columns.tolist())
            print("ClearSkyIndex in DataFrame after creating missing ones:", 'ClearSkyIndex' in df.columns)
        
        # Filter columns that actually exist in the dataframe (should now be all of them)
        minmax_columns = [col for col in minmax_columns if col in df.columns]
        standard_columns = [col for col in standard_columns if col in df.columns]
        
        # Apply scalers with robust error handling and explicit mismatch reporting
        try:
            # Apply MinMaxScaler to radiation data
            if 'GlobalRadiation [W m-2]' in df.columns:
                try:
                    # Use reshape to handle single column
                    rad_data = df['GlobalRadiation [W m-2]'].values.reshape(-1, 1)
                    df['GlobalRadiation [W m-2]'] = self.minmax_scaler.transform(rad_data)
                    print("Normalized GlobalRadiation using MinMaxScaler")
                except Exception as e:
                    print(f"SCALER MISMATCH: Error normalizing GlobalRadiation: {e}")
                    print("WARNING: Using fallback normalization for GlobalRadiation")
                    # Simple fallback normalization
                    max_val = df['GlobalRadiation [W m-2]'].max()
                    if max_val > 0:
                        df['GlobalRadiation [W m-2]'] = df['GlobalRadiation [W m-2]'] / max_val
            
            # Apply StandardScaler to temperature data
            if 'Temperature [degree_Celsius]' in df.columns:
                try:
                    temp_data = df['Temperature [degree_Celsius]'].values.reshape(-1, 1)
                    df['Temperature [degree_Celsius]'] = self.standard_scaler.transform(temp_data)
                    print("Normalized Temperature using StandardScaler")
                except Exception as e:
                    print(f"SCALER MISMATCH: Error normalizing Temperature: {e}")
                    print("WARNING: Using fallback normalization for Temperature")
                    # Simple standardization as fallback
                    mean = df['Temperature [degree_Celsius]'].mean()
                    std = df['Temperature [degree_Celsius]'].std()
                    if std > 0:
                        df['Temperature [degree_Celsius]'] = (df['Temperature [degree_Celsius]'] - mean) / std
            
            # Apply StandardScaler to wind speed data
            if 'WindSpeed [m s-1]' in df.columns:
                try:
                    wind_data = df['WindSpeed [m s-1]'].values.reshape(-1, 1)
                    df['WindSpeed [m s-1]'] = self.standard_scaler.transform(wind_data)
                    print("Normalized WindSpeed using StandardScaler")
                except Exception as e:
                    print(f"SCALER MISMATCH: Error normalizing WindSpeed: {e}")
                    print("WARNING: Using fallback normalization for WindSpeed")
                    # Simple standardization as fallback
                    mean = df['WindSpeed [m s-1]'].mean()
                    std = df['WindSpeed [m s-1]'].std()
                    if std > 0:
                        df['WindSpeed [m s-1]'] = (df['WindSpeed [m s-1]'] - mean) / std
            
            # For ClearSkyIndex, use simple min-max scaling (it's already between 0-1)
            if 'ClearSkyIndex' in df.columns:
                try:
                    # Try to use the standard scaler first
                    csi_data = df['ClearSkyIndex'].values.reshape(-1, 1)
                    df['ClearSkyIndex'] = self.standard_scaler.transform(csi_data)
                    print("Normalized ClearSkyIndex using StandardScaler")
                except Exception as e:
                    print(f"SCALER MISMATCH: Error normalizing ClearSkyIndex: {e}")
                    print("WARNING: ClearSkyIndex will use its original values (already in 0-1 range)")
                    # No need to transform, it's already in the right range
                    
        except Exception as e:
            print(f"CRITICAL SCALER ERROR: {e}")
            print(f"Current columns: {df.columns.tolist()}")
            print("WARNING: Continuing with unnormalized data - prediction accuracy may be affected")
        
        # Final verification of critical features
        if 'ClearSkyIndex' not in df.columns:
            print("WARNING: ClearSkyIndex missing at end of normalization")
        
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
        
        # Verify ClearSkyIndex was created
        if 'ClearSkyIndex' not in df.columns:
            print("WARNING: ClearSkyIndex not created after adding derived features")
        
        # Proactively add all required standard features to ensure they're available
        print("Adding standard feature set required for model inference")
        
        # Add standard features needed for prediction
        try:
            # Radiation data
            if 'GlobalRadiation [W m-2]' not in df.columns:
                if 'poa_global' in df.columns:
                    df['GlobalRadiation [W m-2]'] = df['poa_global']
                    print("Created GlobalRadiation [W m-2] from poa_global")
                else:
                    raise KeyError("Cannot create GlobalRadiation [W m-2] - no poa_global column available")
            
            # Temperature data
            if 'Temperature [degree_Celsius]' not in df.columns:
                if 'temp_air' in df.columns:
                    df['Temperature [degree_Celsius]'] = df['temp_air']
                    print("Created Temperature [degree_Celsius] from temp_air")
                else:
                    raise KeyError("Cannot create Temperature [degree_Celsius] - no temp_air column available")
            
            # Wind data
            if 'WindSpeed [m s-1]' not in df.columns:
                if 'wind_speed' in df.columns:
                    df['WindSpeed [m s-1]'] = df['wind_speed']
                    print("Created WindSpeed [m s-1] from wind_speed")
                else:
                    raise KeyError("Cannot create WindSpeed [m s-1] - no wind_speed column available")
            
            # Clear sky index
            if 'ClearSkyIndex' not in df.columns:
                if 'total_cloud_cover' in df.columns:
                    df['ClearSkyIndex'] = 1.0 - df['total_cloud_cover']
                    print("Created ClearSkyIndex from total_cloud_cover")
                else:
                    print("Warning: total_cloud_cover not available - setting ClearSkyIndex to default value")
                    df['ClearSkyIndex'] = 0.5  # Default value if no cloud cover data
                    
        except KeyError as e:
            print(f"Error creating standard features: {e}")
            print(f"Available columns: {df.columns.tolist()}")
            raise
        
        # Normalize the data
        df = self.normalize_data(df)
        
        # Verify critical features after normalization
        if 'ClearSkyIndex' not in df.columns:
            print("WARNING: ClearSkyIndex missing after normalization")
        
        # Get the standard features set
        features = self.feature_sets
        
        # Extract features as numpy array
        # Check if all features are available
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            print("Creating a DataFrame with just the standard features needed for prediction")
            
            # Debug: Print each missing feature
            for feature in missing_features:
                print(f"Missing feature: {feature}")
                
            # Create a new DataFrame with standardized feature set
            X = np.zeros((len(df), len(features)))
            print(f"Created empty feature array with shape: {X.shape}")
        else:
            # All features are available, use them directly
            X = df[features].values
            print("All features available for prediction array")
        
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
    
    def predict(self, weather_data=None, hours=24, return_df=True):
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
            
            # Verify critical features before prediction
            if 'ClearSkyIndex' not in df.columns:
                print("WARNING: ClearSkyIndex missing before prediction")
            
            # Create a new DataFrame with exactly the required features
            # This ensures the correct order and presence of all features
            training_data = pd.DataFrame()
            
            # Add required features in the correct order, with fallbacks
            if 'GlobalRadiation [W m-2]' in df.columns:
                training_data['GlobalRadiation [W m-2]'] = df['GlobalRadiation [W m-2]']
            elif 'poa_global' in df.columns:
                training_data['GlobalRadiation [W m-2]'] = df['poa_global']
                print("Using poa_global for GlobalRadiation [W m-2]")
            else:
                raise KeyError("No radiation data available for prediction")
                
            if 'Temperature [degree_Celsius]' in df.columns:
                training_data['Temperature [degree_Celsius]'] = df['Temperature [degree_Celsius]']
            elif 'temp_air' in df.columns:
                training_data['Temperature [degree_Celsius]'] = df['temp_air']
                print("Using temp_air for Temperature [degree_Celsius]")
            else:
                raise KeyError("No temperature data available for prediction")
                
            if 'WindSpeed [m s-1]' in df.columns:
                training_data['WindSpeed [m s-1]'] = df['WindSpeed [m s-1]']
            elif 'wind_speed' in df.columns:
                training_data['WindSpeed [m s-1]'] = df['wind_speed']
                print("Using wind_speed for WindSpeed [m s-1]")
            else:
                raise KeyError("No wind speed data available for prediction")
            
            # Fix indentation: ClearSkyIndex check was incorrectly nested inside wind speed else block
            if 'ClearSkyIndex' in df.columns:
                training_data['ClearSkyIndex'] = df['ClearSkyIndex']
            elif 'total_cloud_cover' in df.columns:
                training_data['ClearSkyIndex'] = 1.0 - df['total_cloud_cover']
                print("Using 1.0 - total_cloud_cover for ClearSkyIndex")
            else:
                print("Warning: No cloud cover data available. Using default ClearSkyIndex value.")
                training_data['ClearSkyIndex'] = 0.5  # Default value
                
            # Add time features
            for time_feature in ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'isNight']:
                if time_feature in df.columns:
                    training_data[time_feature] = df[time_feature]
                else:
                    raise KeyError(f"Missing time feature: {time_feature}")
            
            # Our standard feature set should exactly match what we defined in _define_feature_sets
            standard_features = self.feature_sets
            
            # Verify we have all needed features
            missing_features = [f for f in standard_features if f not in training_data.columns]
            if missing_features:
                raise KeyError(f"Missing required features after preparation: {missing_features}")
                
            # Verify features match between model and data
            if set(standard_features) != set(training_data.columns):
                print("WARNING: Feature mismatch between model and prepared data")
                print(f"Model features: {standard_features}")
                print(f"Data features: {list(training_data.columns)}")
            else:
                print("Features match between model and prepared data")
                
            # Use the standardized training data for prediction
            X = training_data[standard_features].values
            print(f"Prepared prediction data with shape: {X.shape}")
            
            # Create sequences
            X_seq = self.create_sequences(X, self.sequence_length)
            
            # Make predictions
            y_pred = self.model.predict(X_seq)
            
            # Inverse transform predictions
            y_pred_inv = self.target_scaler.inverse_transform(y_pred)
            
            # Ensure non-negative power values
            y_pred_inv = np.maximum(y_pred_inv, 0)
            
            print(f"Generated {len(y_pred_inv)} prediction points")
            
        except Exception as e:
            print(f"Error running forecast: {e}")
            raise
        
        if return_df:
            # Create a DataFrame with predictions and features
            # For a proper 24-hour forecast, we should return all hours, not just from sequence_length-1
            # We'll match predictions with the actual hours in the forecast
            
            # IMPORTANT CHANGE: Return all available forecast hours instead of just the sequence tail
            # The first valid prediction is for time at index sequence_length-1
            # pred_start_idx = self.sequence_length - 1
            
            # Instead of limiting output to just a few hours at the end, return all hours with predictions
            pred_df = df.copy()
            
            # We have fewer predictions than input hours due to the sequence requirement
            # Fill the first sequence_length-1 hours with NaN to indicate no predictions
            pred_df['power_w'] = np.nan
            
            # Add predictions to all hours that have them (hours after sequence formation)
            valid_hours = len(pred_df) - self.sequence_length + 1
            if valid_hours > 0:
                pred_df.iloc[self.sequence_length-1:, pred_df.columns.get_loc('power_w')] = y_pred_inv.flatten()
            
            # Reset index to have timestamp as a column
            pred_df.reset_index(inplace=True)
            
            # Print detailed information about the forecast for debugging
            print(f"Total hours in weather data: {len(df)}")
            print(f"Sequence length: {self.sequence_length}")
            print(f"Valid prediction hours: {valid_hours}")
            print(f"Number of predictions: {len(y_pred_inv)}")
            
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
        
        # Display the unnormalized irradiance values
        # Get the real-world irradiance values from the raw data (before normalization)
        if 'poa_global' in pred_df.columns:
            # Use raw values directly as they're already in W/m²
            ax2.plot(pred_df['timestamp'], pred_df['poa_global'], 'r-', label='Global Irradiance (W/m²)')
        else:
            # If we only have normalized values, attempt to unnormalize them
            try:
                # Create a copy of normalized irradiance values
                irradiance_values = pred_df['GlobalRadiation [W m-2]'].values.reshape(-1, 1)
                # Inverse transform to get original scale if possible
                if hasattr(self, 'minmax_scaler'):
                    unnormalized_irradiance = self.minmax_scaler.inverse_transform(irradiance_values)
                    ax2.plot(pred_df['timestamp'], unnormalized_irradiance, 'r-', label='Global Irradiance (W/m²)')
                else:
                    # Fallback - use as is but make it clear it might be normalized
                    ax2.plot(pred_df['timestamp'], pred_df['GlobalRadiation [W m-2]'], 'r-', 
                            label='Global Irradiance (normalized, W/m²)')
            except Exception as e:
                print(f"Warning: Could not unnormalize irradiance values: {e}")
                # Use available irradiance column
                ax2.plot(pred_df['timestamp'], pred_df['GlobalRadiation [W m-2]'], 'r-', 
                        label='Global Irradiance (W/m²)')
                
        ax2.set_ylabel('Irradiance (W/m²)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Add night periods as shaded areas
        if 'isNight' in pred_df.columns:
            night_periods = []
            current_night = None
            
            # Identify continuous night periods
            for i, row in pred_df.iterrows():
                if row['isNight'] == 1 and current_night is None:
                    current_night = i
                elif row['isNight'] == 0 and current_night is not None:
                    night_periods.append((current_night, i))
                    current_night = None
            
            # Add the last night period if it extends to the end
            if current_night is not None:
                night_periods.append((current_night, len(pred_df) - 1))
            
            # Add shading for night periods
            for start, end in night_periods:
                ax1.axvspan(pred_df['timestamp'].iloc[start], pred_df['timestamp'].iloc[end], 
                            alpha=0.2, color='gray')
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.title('PV Power Forecast')
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
    forecaster = PVForecaster(sequence_length=24)
    
    # Make predictions for 24 hours explicitly
    try:
        # Request 24 hours forecast
        print("Generating 24-hour forecast from current time...")
        pred_df = forecaster.predict(hours=24, return_df=True)
        
        # Plot forecast
        forecaster.plot_forecast(pred_df, save_path='results/forecast_plot.png')
        
        # Export to CSV
        forecaster.export_to_csv(pred_df)
        
        # Export to JSON
        forecaster.export_to_json(pred_df)
        
        # Print summary
        print("\nForecast Summary:")
        print(f"Sequence length: {forecaster.sequence_length}")
        print(f"Forecast period: {pred_df['timestamp'].min()} to {pred_df['timestamp'].max()}")
        print(f"Number of forecast hours: {len(pred_df)}")
        print(f"Maximum predicted power: {pred_df['power_w'].max():.2f} W")
        print(f"Average predicted power: {pred_df['power_w'].mean():.2f} W")
        print(f"Total predicted energy: {pred_df['power_w'].sum() / 1000:.2f} kWh")
        
    except Exception as e:
        print(f"Error making predictions: {e}")


if __name__ == "__main__":
    main()
