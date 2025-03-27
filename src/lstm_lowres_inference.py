#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LSTM Model Inference Script for 1-hour Resolution PV Power Forecasting

This script loads a pre-trained LSTM model and performs inference to generate
a 24-hour forecast of PV power output. It works by:
1. Loading the most recent model and corresponding scalers
2. Fetching recent historical data for the initial sequence
3. Getting or generating forecasted weather data for the prediction horizon
4. Calculating derived features and scaling appropriately
5. Performing iterative prediction, one hour at a time
6. Producing output with timestamps and predicted power values

The script supports both direct execution and import as a module.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import joblib
import os
import glob
from datetime import datetime, timedelta
import pvlib
from pvlib.location import Location
from pvlib.clearsky import simplified_solis
import requests
import json
import argparse

class LSTMLowResInference:
    def __init__(self, sequence_length=24):
        """
        Initialize the inference system with model parameters.
        
        Args:
            sequence_length (int): Number of time steps to look back (default: 24 hours)
        """
        self.sequence_length = sequence_length
        
        # Location parameters for Leoben (from forecast.py)
        self.latitude = 47.38770748541585
        self.longitude = 15.094127778561258
        self.altitude = 541
        self.tz = 'Etc/GMT+1'
        
        # Define feature sets
        self.feature_sets = self._define_feature_sets()
        
        # Load model and scalers
        self._load_model_and_scalers()
        
    def _define_feature_sets(self):
        """Define the feature sets used by the model."""
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
        
        # Group features by scaling method
        minmax_features = ['GlobalRadiation [W m-2]', 'ClearSkyIndex']
        standard_features = ['Temperature [degree_Celsius]']
        robust_features = ['WindSpeed [m s-1]']
        no_scaling_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'isNight']
        
        return {
            'all_features': features,
            'minmax_features': minmax_features,
            'standard_features': standard_features,
            'robust_features': robust_features,
            'no_scaling_features': no_scaling_features,
            'target': 'power_w'
        }
        
    def _load_model_and_scalers(self):
        """Load the latest trained model and corresponding scalers."""
        try:
            # Find the latest model file
            model_path = self._find_latest_file('models/lstm_lowres/final_model_*.keras')
            if not model_path:
                model_path = self._find_latest_file('models/lstm_lowres/model_*.keras')
                
            if not model_path:
                raise FileNotFoundError("No LSTM model files found in models/lstm_lowres/")
                
            # Load the model
            self.model = load_model(model_path)
            print(f"Loaded model from {model_path}")
            
            # Extract timestamp from model filename to find matching scalers
            timestamp = os.path.basename(model_path).split('final_model_')[-1].split('.keras')[0]
            if 'final_model_' not in model_path:
                timestamp = os.path.basename(model_path).split('model_')[-1].split('.keras')[0]
                
            print(f"Using model timestamp: {timestamp}")
            
            # Load scalers with matching timestamp
            minmax_path = f'models/lstm_lowres/minmax_scaler_{timestamp}.pkl'
            standard_path = f'models/lstm_lowres/standard_scaler_{timestamp}.pkl'
            robust_path = f'models/lstm_lowres/robust_scaler_{timestamp}.pkl'
            target_path = f'models/lstm_lowres/target_scaler_{timestamp}.pkl'
            
            # Load the scalers
            self.minmax_scaler = joblib.load(minmax_path)
            self.standard_scaler = joblib.load(standard_path)
            self.robust_scaler = joblib.load(robust_path)
            self.target_scaler = joblib.load(target_path)
            
            print("Successfully loaded all scalers")
            
        except Exception as e:
            print(f"Error loading model or scalers: {e}")
            raise
    
    def _find_latest_file(self, pattern):
        """Find the most recent file matching the given pattern."""
        files = glob.glob(pattern)
        if not files:
            return None
        return max(files, key=os.path.getctime)
        
    def load_historical_data(self, data_path='data/station_data_1h.parquet'):
        """
        Load the most recent historical data for initial sequence.
        
        Args:
            data_path: Path to the historical data file
            
        Returns:
            DataFrame with the most recent sequence_length rows
        """
        print(f"Loading historical data from {data_path}...")
        df = pd.read_parquet(data_path)
        
        # Sort by timestamp to ensure we get the latest data
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
            
        # Get the most recent sequence_length hours
        recent_data = df.tail(self.sequence_length)
        
        print(f"Loaded historical data from {recent_data.index.min()} to {recent_data.index.max()}")
        return recent_data
        
    def fetch_weather_forecast(self, hours=24):
        """
        Fetch weather forecast data from the API for the next 24 hours.
        
        Args:
            hours: Number of hours to forecast
            
        Returns:
            dict: The fetched weather forecast data as a dictionary
        """
        lat_lon = f"{self.latitude},{self.longitude}"
        current_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        fc_start_time = current_time.strftime("%Y-%m-%dT%H:%M")
        fc_end_time = (current_time + pd.Timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M")
        
        url = f"https://dataset.api.hub.geosphere.at/v1/timeseries/forecast/nwp-v1-1h-2500m?lat_lon={lat_lon}&parameters=cape&parameters=cin&parameters=grad&parameters=mnt2m&parameters=mxt2m&parameters=rain_acc&parameters=rh2m&parameters=rr_acc&parameters=snow_acc&parameters=snowlmt&parameters=sp&parameters=sundur_acc&parameters=sy&parameters=t2m&parameters=tcc&parameters=u10m&parameters=ugust&parameters=v10m&parameters=vgust&start={fc_start_time}&end={fc_end_time}"
        
        try:
            print(f"Fetching weather forecast from {fc_start_time} to {fc_end_time}...")
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                print("Weather forecast data fetched successfully")
                return data
            else:
                print(f"Failed to fetch data. Status code: {response.status_code}")
                return self._generate_dummy_weather_data(hours)
                
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            print("Falling back to generated dummy forecast data")
            return self._generate_dummy_weather_data(hours)
            
    def process_weather_data(self, data):
        """
        Process the fetched weather forecast data and create a DataFrame.
        
        Args:
            data: Dictionary containing the API response or generated dummy data
            
        Returns:
            DataFrame with processed weather data
        """
        if isinstance(data, dict) and 'timestamps' in data:
            # Process actual API data
            time_list = data['timestamps']
            parameters = data['features'][0]['properties']['parameters']
            
            # Calculate global irradiation difference and scale to W/m²
            global_irradiation = [(parameters['grad']['data'][i + 1] - val) / 3600 for i, val in enumerate(parameters['grad']['data'][:-1])]
            global_irradiation.insert(0, 0)  # Adding 0 as a placeholder
            
            # Create DataFrame with all parameters
            forecast_df = pd.DataFrame({
                'timestamp': pd.to_datetime(time_list),
                'temp_air': parameters['t2m']['data'],
                'wind_speed': np.sqrt(np.array(parameters['u10m']['data'])**2 + np.array(parameters['v10m']['data'])**2),
                'poa_global': global_irradiation,
                'total_cloud_cover': parameters['tcc']['data'],
                'wind_speed_east': parameters['u10m']['data'],
                'wind_speed_north': parameters['v10m']['data']
            })
            
            # Add other parameters if available
            if 'cape' in parameters:
                forecast_df['cape'] = parameters['cape']['data']
            if 'rh2m' in parameters:
                forecast_df['rel_humidity'] = parameters['rh2m']['data']
                
        else:
            # Process dummy data that's already a DataFrame
            forecast_df = data
        
        # Set timestamp as index
        if 'timestamp' in forecast_df.columns:
            forecast_df.set_index('timestamp', inplace=True)
        
        # Create essential column mappings to match model expectations
        if 'GlobalRadiation [W m-2]' not in forecast_df.columns and 'poa_global' in forecast_df.columns:
            forecast_df['GlobalRadiation [W m-2]'] = forecast_df['poa_global']
        
        if 'Temperature [degree_Celsius]' not in forecast_df.columns and 'temp_air' in forecast_df.columns:
            forecast_df['Temperature [degree_Celsius]'] = forecast_df['temp_air']
        
        if 'WindSpeed [m s-1]' not in forecast_df.columns and 'wind_speed' in forecast_df.columns:
            forecast_df['WindSpeed [m s-1]'] = forecast_df['wind_speed']
        
        return forecast_df
    
    def _generate_dummy_weather_data(self, hours=24):
        """
        Generate dummy weather data when API is unavailable.
        
        Args:
            hours: Number of hours to forecast
            
        Returns:
            DataFrame with generated weather data
        """
        print("Generating synthetic weather forecast data...")
        current_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        timestamps = [current_time + timedelta(hours=i) for i in range(hours)]
        
        # Generate synthetic data with reasonable patterns
        forecast_df = pd.DataFrame()
        forecast_df['timestamp'] = timestamps
        
        # Hour of day for each timestamp (0-23)
        hours_of_day = [t.hour for t in timestamps]
        
        # Day of year for each timestamp (1-365)
        days_of_year = [t.timetuple().tm_yday for t in timestamps]
        
        # Generate radiation with day/night pattern (peak at noon)
        max_radiation = 800  # W/m²
        radiation = []
        for hour in hours_of_day:
            if 6 <= hour <= 18:  # Daytime hours
                # Sine wave with peak at noon
                rad = max_radiation * np.sin(np.pi * (hour - 6) / 12)
                radiation.append(max(0, rad))
            else:
                radiation.append(0)  # No radiation at night
        
        # Temperature with day/night pattern
        min_temp = 10  # °C
        max_temp = 25  # °C
        temperature = []
        for hour in hours_of_day:
            # Sine wave with minimum at 3am, maximum at 3pm
            phase = (hour - 3) % 24
            temp = min_temp + (max_temp - min_temp) * np.sin(np.pi * phase / 12) ** 2
            temperature.append(temp)
        
        # Wind speed (random with reasonable values)
        wind_speed = np.random.uniform(1, 8, hours)  # m/s
        
        # Wind components (east/north)
        wind_angle = np.random.uniform(0, 2*np.pi, hours)
        wind_east = wind_speed * np.cos(wind_angle)
        wind_north = wind_speed * np.sin(wind_angle)
        
        # Cloud cover (random with reasonable values)
        cloud_cover = np.random.uniform(0, 1, hours)
        
        # Add to DataFrame
        forecast_df['poa_global'] = radiation
        forecast_df['temp_air'] = temperature
        forecast_df['wind_speed'] = wind_speed
        forecast_df['wind_speed_east'] = wind_east
        forecast_df['wind_speed_north'] = wind_north
        forecast_df['total_cloud_cover'] = cloud_cover
        
        return forecast_df
    
    def calculate_derived_features(self, df):
        """
        Calculate derived time-based features for the data.
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            DataFrame with added derived features
        """
        # Hour of day features
        df['hour'] = df.index.hour + df.index.minute / 60
        angle_hour = 2 * np.pi * df['hour'] / 24
        df['hour_sin'] = np.sin(angle_hour)
        df['hour_cos'] = np.cos(angle_hour)
        
        # Day of year features
        df['day_of_year'] = df.index.dayofyear
        angle_day = 2 * np.pi * df.index.dayofyear / 365
        df['day_sin'] = np.sin(angle_day)
        df['day_cos'] = np.cos(angle_day)
        
        # Calculate ClearSkyIndex and isNight
        df = self._calculate_clear_sky_and_night(df)
        
        return df
    
    def _calculate_clear_sky_and_night(self, df):
        """
        Calculate ClearSkyIndex and isNight features.
        
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
        
        # Calculate ClearSkyIndex from total_cloud_cover if available
        if 'total_cloud_cover' in df.columns:
            # Inverse relationship: 0 cloud cover = 1.0 clear sky index
            df['ClearSkyIndex'] = 1.0 - df['total_cloud_cover']
        else:
            # Create dummy ClearSkyIndex based on time of day if cloud cover data isn't available
            df['ClearSkyIndex'] = (~night_mask).astype(float) * 0.8  # Assumed mostly clear during day
        
        return df
    
    def scale_features(self, df):
        """
        Scale features using the trained scalers.
        
        Args:
            df: DataFrame with all features
            
        Returns:
            DataFrame with scaled features
        """
        # Create a copy of the DataFrame to avoid modifying the original
        scaled_df = df.copy()
        
        # Group features by scaling method as they were during training
        minmax_features = [f for f in self.feature_sets['minmax_features'] if f in df.columns]
        standard_features = [f for f in self.feature_sets['standard_features'] if f in df.columns]
        robust_features = [f for f in self.feature_sets['robust_features'] if f in df.columns]
        
        # Apply MinMaxScaler to minmax_features group (if any exist)
        if minmax_features:
            # When a scaler was trained on multiple columns, we need to transform them together
            scaled_values = self.minmax_scaler.transform(df[minmax_features])
            # Update the DataFrame with scaled values
            for i, feature in enumerate(minmax_features):
                scaled_df[feature] = scaled_values[:, i]
        
        # Apply StandardScaler to standard_features group (if any exist)
        if standard_features:
            scaled_values = self.standard_scaler.transform(df[standard_features])
            for i, feature in enumerate(standard_features):
                scaled_df[feature] = scaled_values[:, i]
        
        # Apply RobustScaler to robust_features group (if any exist)
        if robust_features:
            scaled_values = self.robust_scaler.transform(df[robust_features])
            for i, feature in enumerate(robust_features):
                scaled_df[feature] = scaled_values[:, i]
        
        # No scaling needed for time-based features
        # They are already in appropriate ranges
        
        # Ensure all required features are present and in the correct order
        available_features = [f for f in self.feature_sets['all_features'] if f in scaled_df.columns]
        return scaled_df[available_features]
    
    def create_sequence(self, df):
        """
        Create a sequence for LSTM input.
        
        Args:
            df: DataFrame with features
            
        Returns:
            numpy array with shape (1, sequence_length, num_features)
        """
        # Convert DataFrame to numpy array
        sequence = df.values
        
        # Reshape to (1, sequence_length, num_features)
        return sequence.reshape(1, sequence.shape[0], sequence.shape[1])
    
    def predict_next_24h(self):
        """
        Predict PV power output for the next 24 hours.
        
        Returns:
            DataFrame with timestamp and predicted power
        """
        print("\nStarting 24-hour PV power forecast...")
        
        # 1. Load historical data for initial sequence
        historical_data = self.load_historical_data()
        
        # 2. Fetch raw weather forecast data
        raw_forecast_data = self.fetch_weather_forecast(hours=24)
        
        # 3. Process the raw weather data into a DataFrame
        forecast_data = self.process_weather_data(raw_forecast_data)
        print(f"Processed forecast data with shape: {forecast_data.shape}")
        
        # Print column information for debugging
        print("\nHistorical data columns:", historical_data.columns.tolist())
        
        # 4. Calculate derived features for historical data
        print("Adding derived features to historical data...")
        historical_data = self.calculate_derived_features(historical_data)
        
        # Check that we have all necessary features in the historical data
        missing_features = [f for f in self.feature_sets['all_features'] if f not in historical_data.columns]
        if missing_features:
            print(f"Warning: Missing features in historical data: {missing_features}")
            # Try to create missing features or use default values
            if 'ClearSkyIndex' in missing_features and 'total_cloud_cover' in historical_data.columns:
                historical_data['ClearSkyIndex'] = 1.0 - historical_data['total_cloud_cover']
                missing_features.remove('ClearSkyIndex')
            # For any remaining missing features, use zeros as a fallback
            for feature in missing_features:
                print(f"Creating missing feature with zeros: {feature}")
                historical_data[feature] = 0.0
        
        # 5. Calculate derived features for forecast data
        print("Adding derived features to forecast data...")
        forecast_data = self.calculate_derived_features(forecast_data)
        
        # 6. Set start time to current time (rounded to hour)
        current_date = datetime.now().replace(minute=0, second=0, microsecond=0)
        # Use current time minus sequence length hours as the start reference
        start_time = current_date - timedelta(hours=1)
        print(f"Forecast start time: {start_time}")
        
        # 7. Initialize storage for forecast results
        forecast_results = []
        
        # 8. Prepare initial input sequence from historical data
        current_input = self.scale_features(historical_data)
        print(f"Input sequence prepared with shape: {current_input.shape}")
        
        # 9. Start iterative prediction
        print("\nPerforming iterative prediction for next 24 hours...")
        for i in range(24):
            # Get current forecast timestamp
            current_time = start_time + timedelta(hours=i+1)
            print(f"Predicting for: {current_time}")
            
            # Create sequence for model input
            sequence = self.create_sequence(current_input)
            
            # Predict with the model
            scaled_prediction = self.model.predict(sequence, verbose=0)[0][0]
            print(f"Raw scaled prediction: {scaled_prediction:.4f}")
            
            # Inverse scale the prediction
            power_prediction = self.target_scaler.inverse_transform([[scaled_prediction]])[0][0]
            print(f"Unscaled prediction: {power_prediction:.2f} W")
            
            # Ensure non-negative power prediction
            power_prediction = max(0, power_prediction)
            
            # Check if it's night time and adjust prediction if needed
            is_night = False
            if 'isNight' in current_input.columns:
                current_hour_night_value = current_input['isNight'].iloc[-1]
                if current_hour_night_value > 0.5:  # Using 0.5 as threshold for binary features
                    is_night = True
                    print(f"Night time detected, setting power to 0")
                    power_prediction = 0.0
            
            # Store the result
            forecast_results.append({
                'timestamp': current_time,
                'power_w': power_prediction
            })
            
            # Prepare for next iteration - drop oldest row and add new forecast row
            if i < 23:  # No need to update for the last iteration
                # Get the new forecast row data for the next time step
                next_time = current_time + timedelta(hours=1)
                print(f"Looking for forecast data at: {next_time}")
                next_forecast_row = forecast_data.loc[forecast_data.index == next_time]
                
                if not next_forecast_row.empty:
                    print(f"Found forecast data for {next_time}")
                    # Scale the next forecast row
                    next_forecast_row = self.scale_features(next_forecast_row)
                    
                    # Create new input by removing oldest and adding new forecasted row
                    current_input = current_input.iloc[1:].copy()
                    current_input = pd.concat([current_input, next_forecast_row])
                else:
                    print(f"WARNING: No forecast data found for {next_time}. Using estimate.")
                    # If next hour data isn't available, copy the previous hour with adjustments
                    next_forecast = current_input.iloc[[-1]].copy()
                    next_forecast.index = [next_time]
                    
                    # Adjust hour-based features
                    if 'hour_sin' in next_forecast.columns and 'hour_cos' in next_forecast.columns:
                        angle_hour = 2 * np.pi * next_time.hour / 24
                        next_forecast['hour_sin'] = np.sin(angle_hour)
                        next_forecast['hour_cos'] = np.cos(angle_hour)
                    
                    # Add to current input
                    current_input = current_input.iloc[1:].copy()
                    current_input = pd.concat([current_input, next_forecast])
        
        # 9. Create DataFrame with results
        results_df = pd.DataFrame(forecast_results)
        results_df.set_index('timestamp', inplace=True)
        
        print("\nForecast complete!")
        return results_df
    
    def display_forecast(self, forecast_df):
        """
        Display the forecasted power output.
        
        Args:
            forecast_df: DataFrame with timestamp index and power_w column
        """
        print("\n=== 24-Hour PV Power Forecast ===")
        print("\nTimestamp                   Power (W)")
        print("----------------------------------------")
        
        for idx, row in forecast_df.iterrows():
            timestamp_str = idx.strftime('%Y-%m-%d %H:%M')
            power_str = f"{row['power_w']:.2f}"
            print(f"{timestamp_str}          {power_str:>10}")
        
        # Calculate some statistics
        total_energy = forecast_df['power_w'].sum() / 1000  # kWh
        max_power = forecast_df['power_w'].max()
        avg_power = forecast_df['power_w'].mean()
        
        print("\n=== Summary Statistics ===")
        print(f"Total Energy:  {total_energy:.2f} kWh")
        print(f"Maximum Power: {max_power:.2f} W")
        print(f"Average Power: {avg_power:.2f} W")
    
    def plot_forecast(self, forecast_df, save_path=None):
        """
        Plot the forecasted power output.
        
        Args:
            forecast_df: DataFrame with timestamp index and power_w column
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(12, 6))
        
        # Plot power output
        plt.plot(forecast_df.index, forecast_df['power_w'], 'b-', linewidth=2, label='Predicted Power')
        
        # Add shading for night hours
        for i in range(len(forecast_df)-1):
            if forecast_df.index[i].hour >= 18 or forecast_df.index[i].hour < 6:
                plt.axvspan(forecast_df.index[i], forecast_df.index[i+1], 
                            alpha=0.2, color='gray')
        
        # Format plot
        plt.title('24-Hour PV Power Forecast', fontsize=16)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Power (W)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper left')
        
        # Format x-axis with nicer time labels
        plt.gcf().autofmt_xdate()
        
        # Show/save plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function to run the inference script."""
    parser = argparse.ArgumentParser(description='LSTM PV Power Forecasting')
    parser.add_argument('--plot', action='store_true', help='Plot the forecast results')
    parser.add_argument('--save-plot', type=str, help='Save the plot to the specified path')
    args = parser.parse_args()
    
    # Create inference object
    inference = LSTMLowResInference()
    
    # Generate forecast
    forecast_df = inference.predict_next_24h()
    
    # Display the forecast
    inference.display_forecast(forecast_df)
    
    # Plot the forecast if requested
    if args.plot or args.save_plot:
        inference.plot_forecast(forecast_df, args.save_plot)
    
    return forecast_df

if __name__ == "__main__":
    forecast_df = main()