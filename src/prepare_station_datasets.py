import pandas as pd
import numpy as np
from datetime import datetime
import os
import pyarrow as pa
import pyarrow.parquet as pq
import pvlib
from pvlib.location import Location
from pvlib.clearsky import simplified_solis
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib

class StationDataProcessor:
    def __init__(self, chunk_size: int = 100000):
        """
        Initialize the data processor with configuration.
        """
        self.chunk_size = chunk_size
        self.high_res_frequency = '10min'  # Original station data frequency
        self.low_res_frequency = '1h'      # Target low resolution frequency
        self.minmax_scaler = None
        self.standard_scaler = None
        
    def read_csv_in_chunks(self, filepath: str, date_column: str, 
                          date_format: str = '%Y-%m-%dT%H:%M+00:00', 
                          skiprows: int = 0, sep: str = ',',
                          decimal: str = '.', encoding: str = 'utf-8') -> pd.DataFrame:
        """
        Read CSV file in chunks and process each chunk.
        """
        chunks = []
        # Read the file with a single separator to get the combined columns
        for chunk in pd.read_csv(filepath, sep=sep, chunksize=self.chunk_size, 
                               skiprows=skiprows, decimal=decimal, 
                               encoding=encoding):
            # Convert datetime
            chunk[date_column] = pd.to_datetime(chunk[date_column], format=date_format)
            
            # Set datetime as index
            chunk.set_index(date_column, inplace=True)
            chunks.append(chunk)
            
        return pd.concat(chunks)

    def process_station_data(self, filepath: str) -> pd.DataFrame:
        """Process local weather station data (10min) with quality filtering."""
        print(f"Reading station data from {filepath}...")
        df = self.read_csv_in_chunks(
            filepath=filepath,
            date_column='time',
            date_format='%Y-%m-%dT%H:%M+00:00'
        )
        
        print(f"Raw station data shape: {df.shape}")
        print("Sample of raw data:")
        print(df.head())
        
        # Define critical measurements and their quality flags with descriptive names
        critical_columns = {
            'cglo': 'cglo_flag',  # Global radiation
            'tl': 'tl_flag',      # Air temperature
            'ff': 'ff_flag'       # Wind speed
        }
        
        # Filter based on quality flags (flag value 12 seems to be good based on data inspection)
        for col, flag_col in critical_columns.items():
            if col in df.columns and flag_col in df.columns:
                df = df[df[flag_col] == 12]
        
        # Select relevant columns (excluding flag columns)
        relevant_cols = ['cglo', 'tl', 'ff', 'rr', 'p']
        available_cols = [col for col in relevant_cols if col in df.columns]
        df = df[available_cols]
        
        # Rename columns to be more descriptive
        column_mapping = {
            'cglo': 'GlobalRadiation [W m-2]',  # Global radiation measurement
            'tl': 'Temperature [degree_Celsius]',  # Air temperature
            'ff': 'WindSpeed [m s-1]',  # Wind speed
            'rr': 'Precipitation [mm]',  # Rainfall/precipitation
            'p': 'Pressure [hPa]'  # Air pressure
        }
        df = df.rename(columns=column_mapping)
        
        print(f"Processed station data shape: {df.shape}")
        print("Sample of processed data:")
        print(df.head())
        
        return df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features for machine learning."""
        # Calculate hour with resolution based on the data frequency
        df['hour'] = df.index.hour + df.index.minute / 60
        df['day_of_year'] = df.index.dayofyear
        
        # Add circular encoding for hour (sin/cos)
        angle_hour = 2 * np.pi * df['hour'] / 24
        df['hour_sin'] = np.sin(angle_hour)
        df['hour_cos'] = np.cos(angle_hour)
        
        # Add circular encoding for day of year (sin/cos)
        angle_day = 2 * np.pi * df['day_of_year'] / 365.25  # Account for leap years
        df['day_sin'] = np.sin(angle_day)
        df['day_cos'] = np.cos(angle_day)
        
        return df
    
    def add_night_mask(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add isNight mask based on irradiation values.
        We consider it night when global radiation is below a threshold.
        """
        # Define threshold for night (very low irradiation)
        NIGHT_THRESHOLD = 10  # W/m²
        
        # Create isNight mask (1 for night, 0 for day)
        if 'GlobalRadiation [W m-2]' in df.columns:
            df['isNight'] = (df['GlobalRadiation [W m-2]'] < NIGHT_THRESHOLD).astype(int)
        else:
            print("Warning: GlobalRadiation column not found. Cannot create isNight mask based on irradiation.")
            # As a fallback, calculate isNight based on hour (approximate)
            df['isNight'] = ((df.index.hour >= 19) | (df.index.hour < 6)).astype(int)
        
        return df
    
    def calculate_clear_sky(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate clear sky irradiance values using pvlib."""
        # Define location parameters for Leoben
        latitude = 47.38770748541585
        longitude = 15.094127778561258
        altitude = 541  # approximate altitude for Leoben
        tz = 'Etc/GMT+1'
        location = Location(latitude, longitude, tz, altitude)

        # Get solar position
        solar_position = location.get_solarposition(df.index)
        
        # Calculate pressure based on altitude (approximately) and convert from hPa to Pa
        pressure = pvlib.atmosphere.alt2pres(altitude) * 100  # Convert from hPa to Pa
        
        # Use a default value for precipitable water
        precipitable_water = 1.0  # Default value as a simple approximation

        # Calculate apparent elevation and ensure it's not negative
        apparent_elevation = 90 - solar_position['apparent_zenith']
        apparent_elevation = apparent_elevation.clip(lower=0)  # Set negative elevations to 0
        
        # Calculate clear sky irradiance using simplified Solis model
        clear_sky = simplified_solis(
            apparent_elevation=apparent_elevation,
            aod700=0.1,  # Default value for aerosol optical depth
            precipitable_water=precipitable_water,
            pressure=pressure
        )
        
        # Set irradiance components to 0 for nighttime (elevation <= 0)
        night_mask = (apparent_elevation <= 0)
        clear_sky['ghi'][night_mask] = 0
        clear_sky['dni'][night_mask] = 0
        clear_sky['dhi'][night_mask] = 0
        
        # Add isNight feature based on solar position (1 for night, 0 for day)
        df['isNight_solar'] = night_mask.astype(int)
        
        # Add clear sky values to dataframe
        df['ClearSkyGHI'] = clear_sky['ghi']
        df['ClearSkyDNI'] = clear_sky['dni']
        df['ClearSkyDHI'] = clear_sky['dhi']
        
        # Calculate clear sky index (ratio of measured to clear sky GHI)
        if 'GlobalRadiation [W m-2]' in df.columns:
            # Handle division by zero and very small values
            df['ClearSkyIndex'] = np.where(
                df['ClearSkyGHI'] > 10,  # Only calculate for significant irradiance
                df['GlobalRadiation [W m-2]'] / df['ClearSkyGHI'],
                0  # Set to 0 for nighttime or very low irradiance
            )
            
            # Clip unrealistic clear sky index values
            df['ClearSkyIndex'] = df['ClearSkyIndex'].clip(0, 1.5)  # Typical range is 0-1.2
        
        return df
    
    def process_pv_data(self, filepath: str) -> pd.DataFrame:
        """Process PV production data."""
        print(f"\nReading PV data from {filepath}...")
        
        # First read just the header to get column names
        header_df = pd.read_csv(filepath, sep=';', nrows=0)
        print("Column names in file:", header_df.columns.tolist())
        
        # Read the data, skipping the units row
        df = pd.read_csv(filepath, sep=';', skiprows=1, decimal=',',
                        thousands=None, na_values=['n/a', '#WERT!'],
                        encoding='utf-8')
        
        print("First few rows of raw PV data:")
        print(df.head())
        
        # Drop the units row (contains [dd.MM.yyyy HH:mm], [Wh], etc.)
        df = df[~df.iloc[:, 0].str.contains(r'\[.*\]', na=False)]
        
        # Strip any whitespace from values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()
                
        # Get the date column name
        date_col = df.columns[0]  # First column should be the date
        print(f"\nUsing {date_col} as date column")
        
        # Parse timestamps as UTC+1 (local time for Austria/Leoben)
        # Create a new datetime column instead of modifying the original
        df['datetime'] = pd.to_datetime(df[date_col], format='%d.%m.%Y %H:%M')
        
        # Explicitly set timezone to UTC+1
        local_tz = 'Etc/GMT-1'  # Note: Etc/GMT-1 is actually UTC+1 (the sign is inverted in Etc/GMT)
        df['datetime'] = df['datetime'].dt.tz_localize(local_tz)
        
        # Convert to UTC to match station data
        df['datetime'] = df['datetime'].dt.tz_convert('UTC')
        
        # Remove timezone info to match station data format
        df['datetime'] = df['datetime'].dt.tz_localize(None)
        
        print(f"Converted PV timestamps from local time (UTC+1) to UTC")
        
        # Set datetime as index
        df.set_index('datetime', inplace=True)
        
        # Check the columns we have
        print(f"Columns after setting index: {df.columns.tolist()}")
        
        # Rename columns for consistency - make sure we match the number of columns
        if len(df.columns) == 4:  # Including the original date column
            df.columns = ['date_col', 'energy_wh', 'energy_interval', 'power_w']
            # Drop the original date column if it's still there
            if 'date_col' in df.columns:
                df = df.drop(columns=['date_col'])
        elif len(df.columns) == 3:
            df.columns = ['energy_wh', 'energy_interval', 'power_w']
        
        # Convert to numeric, coercing errors to NaN
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print("\nProcessed PV data shape:", df.shape)
        print("Sample of processed PV data:")
        print(df.head())
        
        return df
    
    def save_to_parquet(self, df: pd.DataFrame, output_path: str):
        """Save processed data to parquet format with compression."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert to PyArrow table and save
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_path, compression='snappy')
        print(f"Data saved to {output_path}")
    
    def process_and_save(self, station_path: str, pv_path: str, high_res_output: str, low_res_output: str):
        """
        Process station data and save both high-resolution and low-resolution datasets.
        
        Args:
            station_path: Path to the station data CSV file
            pv_path: Path to the PV data CSV file
            high_res_output: Path to save the high-resolution dataset (10min)
            low_res_output: Path to save the low-resolution dataset (1h)
        """
        # Process station data (10min resolution)
        station_df = self.process_station_data(station_path)
        
        # Calculate clear sky values and add solar-based night mask
        print("Calculating clear sky values...")
        station_df = self.calculate_clear_sky(station_df)
        
        # Add irradiation-based night mask
        print("Adding night mask based on irradiation...")
        station_df = self.add_night_mask(station_df)
        
        # Combine night masks if both are available
        if 'isNight' in station_df.columns and 'isNight_solar' in station_df.columns:
            # Use logical OR - if either method says it's night, consider it night
            station_df['isNight'] = ((station_df['isNight'] == 1) | (station_df['isNight_solar'] == 1)).astype(int)
            station_df = station_df.drop('isNight_solar', axis=1)
        
        # Process PV data
        pv_df = self.process_pv_data(pv_path)
        
        # Define the start timestamp (14.07.2022 19:55)
        start_timestamp = pd.Timestamp('2022-07-14 19:55:00')
        print(f"\nFiltering data to start from {start_timestamp}")
        
        # Filter station data to start from the specified timestamp
        station_df = station_df[station_df.index >= start_timestamp]
        print(f"Filtered station data shape: {station_df.shape}")
        print(f"Station data starts at: {station_df.index.min()}")
        
        # Filter PV data to start from the specified timestamp
        pv_df = pv_df[pv_df.index >= start_timestamp]
        print(f"Filtered PV data shape: {pv_df.shape}")
        print(f"PV data starts at: {pv_df.index.min()}")
        
        # Verify that both datasets start at the same timestamp
        if station_df.index.min() != pv_df.index.min():
            print(f"WARNING: Start timestamps don't match exactly!")
            print(f"Station data starts at: {station_df.index.min()}")
            print(f"PV data starts at: {pv_df.index.min()}")
            time_diff = abs((station_df.index.min() - pv_df.index.min()).total_seconds() / 60)
            print(f"Time difference: {time_diff} minutes")
        else:
            print("Both datasets start at the same timestamp. Good!")
        
        # Merge station data with PV data
        print("\nMerging station data with PV data...")
        print(f"Station data shape: {station_df.shape}")
        print(f"PV data shape: {pv_df.shape}")
        
        # Resample PV data to 10min to match station data frequency
        print("Resampling PV data to 10min frequency...")
        pv_df_resampled = pv_df.resample(self.high_res_frequency).mean()
        print(f"Resampled PV data shape: {pv_df_resampled.shape}")
        
        # Check if timestamps match before joining
        print("Checking if timestamps match before joining...")
        station_timestamps = set(station_df.index)
        pv_timestamps = set(pv_df_resampled.index)
        
        # Find common timestamps
        common_timestamps = station_timestamps.intersection(pv_timestamps)
        print(f"Number of common timestamps: {len(common_timestamps)}")
        
        # Find missing timestamps in each dataset
        missing_in_station = pv_timestamps - station_timestamps
        missing_in_pv = station_timestamps - pv_timestamps
        
        if missing_in_station:
            print(f"Warning: {len(missing_in_station)} timestamps in PV data are missing in station data")
            print(f"First few missing timestamps in station data: {sorted(list(missing_in_station))[:5]}")
        
        if missing_in_pv:
            print(f"Warning: {len(missing_in_pv)} timestamps in station data are missing in PV data")
            print(f"First few missing timestamps in PV data: {sorted(list(missing_in_pv))[:5]}")
        
        # Merge on datetime index (using left join to preserve all station data timestamps)
        merged_df = pd.merge(station_df, pv_df_resampled, left_index=True, right_index=True, how='left')
        print(f"Merged data shape (left join): {merged_df.shape}")
        
        # Handle missing values in target columns
        print("Handling missing values in target columns...")
        target_cols = ['power_w', 'energy_wh', 'energy_interval']
        for col in target_cols:
            if col in merged_df.columns:
                # First, where it's night and target is missing, set to 0
                night_mask = merged_df['isNight'] == 1
                missing_mask = merged_df[col].isna()
                merged_df.loc[night_mask & missing_mask, col] = 0.0
                
                # Count remaining missing values after night-time filling
                remaining_missing = merged_df[col].isna().sum()
                if remaining_missing > 0:
                    print(f"Found {remaining_missing} missing values in {col} after night-time filling")
                    
                    # For remaining missing values, fill with 0 for power and energy
                    # This is a reasonable assumption since missing PV data likely means no production
                    print(f"Filling remaining missing values in {col} with 0")
                    merged_df[col].fillna(0.0, inplace=True)
        
        # Add time-based features to high-resolution data
        print("Adding time-based features to high-resolution data...")
        high_res_df = self.add_time_features(merged_df)
        
        # Save high-resolution dataset (10min)
        print(f"Saving high-resolution (10min) dataset...")
        print(f"High-resolution dataset shape: {high_res_df.shape}")
        print("Sample of high-resolution data:")
        print(high_res_df.head())
        self.save_to_parquet(high_res_df, high_res_output)
        
        # Create low-resolution dataset by resampling to hourly
        print("Creating low-resolution (1h) dataset by resampling...")
        
        # Define aggregation methods for each column
        agg_dict = {
            'GlobalRadiation [W m-2]': 'mean',
            'Temperature [degree_Celsius]': 'mean',
            'WindSpeed [m s-1]': 'mean',
            'Precipitation [mm]': 'sum',  # Sum precipitation over the hour
            'Pressure [hPa]': 'mean',
            'ClearSkyGHI': 'mean',
            'ClearSkyDNI': 'mean',
            'ClearSkyDHI': 'mean',
            'ClearSkyIndex': 'mean',
            'isNight': 'max',  # If any 10min period in the hour is night, consider the hour as night
            'energy_wh': 'last',  # Cumulative value
            'energy_interval': 'sum',  # Sum up interval energy
            'power_w': 'mean'  # Average power
        }
        
        # Filter to include only columns that exist in the dataframe
        agg_dict = {k: v for k, v in agg_dict.items() if k in high_res_df.columns}
        
        # Resample to hourly frequency
        low_res_df = high_res_df.resample(self.low_res_frequency).agg(agg_dict)
        
        # Add time-based features to low-resolution data
        print("Adding time-based features to low-resolution data...")
        low_res_df = self.add_time_features(low_res_df)
        
        # Handle any missing values in the low-resolution dataset
        print("Checking for missing values in low-resolution dataset...")
        target_cols = ['power_w', 'energy_wh', 'energy_interval']
        for col in target_cols:
            if col in low_res_df.columns:
                missing_count = low_res_df[col].isna().sum()
                if missing_count > 0:
                    print(f"Found {missing_count} missing values in {col} in low-resolution dataset")
                    # Fill missing values with 0
                    print(f"Filling missing values in {col} with 0")
                    low_res_df[col] = low_res_df[col].fillna(0.0)
        
        # Save low-resolution dataset (1h)
        print(f"Saving low-resolution (1h) dataset...")
        print(f"Low-resolution dataset shape: {low_res_df.shape}")
        print("Sample of low-resolution data:")
        print(low_res_df.head())
        self.save_to_parquet(low_res_df, low_res_output)
        
        print("Processing complete!")
        return high_res_df, low_res_df

if __name__ == "__main__":
    # File paths
    STATION_PATH = "data/station_data_leoben/Messstationen Zehnminutendaten v2 Datensatz_20220713T0000_20250304T0000.csv"
    PV_PATH = "data/raw_data_evt_act/merge.CSV"
    HIGH_RES_OUTPUT = "data/station_data_10min.parquet"
    LOW_RES_OUTPUT = "data/station_data_1h.parquet"
    
    # Initialize and run processor
    processor = StationDataProcessor(chunk_size=100000)
    high_res_df, low_res_df = processor.process_and_save(STATION_PATH, PV_PATH, HIGH_RES_OUTPUT, LOW_RES_OUTPUT)