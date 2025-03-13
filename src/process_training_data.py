import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import Dict, List, Optional
import pyarrow as pa
import pyarrow.parquet as pq
import pvlib
from pvlib.location import Location
from pvlib.clearsky import simplified_solis
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib

class DataProcessor:
    def __init__(self, chunk_size: int = 100000):
        """
        Initialize the data processor with configuration.
        """
        self.chunk_size = chunk_size
        self.target_frequency = '15min'
        
    def read_csv_in_chunks(self, filepath: str, date_column: str, 
                          date_format: Optional[str] = None, 
                          skiprows: int = 0, sep: str = ',',
                          decimal: str = '.', encoding: str = 'utf-8') -> pd.DataFrame:
        """
        Read CSV file in chunks and process each chunk.
        """
        chunks = []
        # First read the file with a single separator to get the combined columns
        for chunk in pd.read_csv(filepath, sep=sep, chunksize=self.chunk_size, 
                               skiprows=skiprows, decimal=decimal, 
                               encoding=encoding):
            # Convert datetime
            if date_format:
                chunk[date_column] = pd.to_datetime(chunk[date_column], 
                                                  format=date_format)
            else:
                chunk[date_column] = pd.to_datetime(chunk[date_column])
            
            # Set datetime as index
            chunk.set_index(date_column, inplace=True)
            chunks.append(chunk)
            
        return pd.concat(chunks)

    def process_inca_data(self, filepath: str) -> pd.DataFrame:
        """Process INCA weather data (hourly)."""
        df = self.read_csv_in_chunks(
            filepath=filepath,
            date_column='time',
            date_format='%Y-%m-%dT%H:%M+00:00'
        )
        
        # Select relevant columns and add INCA prefix
        columns = ['GL [W m-2]', 'T2M [degree_Celsius]', 'RH2M [percent]',
                  'UU [m s-1]', 'VV [m s-1]']
        df = df[columns]
        
        # Calculate total wind speed from east and north components
        df['INCA_WindSpeed [m s-1]'] = np.sqrt(df['UU [m s-1]']**2 + df['VV [m s-1]']**2)
        
        # Rename columns with INCA prefix
        df = df.rename(columns={
            'GL [W m-2]': 'INCA_GlobalRadiation [W m-2]',
            'T2M [degree_Celsius]': 'INCA_Temperature [degree_Celsius]',
            'RH2M [percent]': 'INCA_RelativeHumidity [percent]',
            'UU [m s-1]': 'INCA_WindU [m s-1]',
            'VV [m s-1]': 'INCA_WindV [m s-1]'
        })
        
        # Resample to target frequency using appropriate methods
        resampled = df.resample(self.target_frequency).interpolate(method='linear')
        
        # Calculate clear sky values
        resampled = self.calculate_clear_sky(resampled)
        
        return resampled

    def process_station_data(self, filepath: str) -> pd.DataFrame:
        """Process local weather station data (10min) with quality filtering."""
        df = self.read_csv_in_chunks(
            filepath=filepath,
            date_column='time',
            date_format='%Y-%m-%dT%H:%M+00:00'
        )
        
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
        
        # Rename columns to be more descriptive with Station prefix
        column_mapping = {
            'cglo': 'Station_GlobalRadiation [W m-2]',  # Global radiation measurement
            'tl': 'Station_Temperature [degree_Celsius]',  # Air temperature
            'ff': 'Station_WindSpeed [m s-1]',  # Wind speed
            'rr': 'Station_Precipitation [mm]',  # Rainfall/precipitation
            'p': 'Station_Pressure [hPa]'  # Air pressure
        }
        df = df.rename(columns=column_mapping)
        
        # Update available columns with new names
        available_cols = [column_mapping[col] for col in available_cols if col in column_mapping]
        
        # Resample to target frequency with renamed columns
        agg_dict = {
            'Station_GlobalRadiation [W m-2]': 'mean',
            'Station_Temperature [degree_Celsius]': 'mean',
            'Station_WindSpeed [m s-1]': 'mean',
            'Station_Precipitation [mm]': 'sum',
            'Station_Pressure [hPa]': 'mean'
        }
        # Only include columns that are actually present
        agg_dict = {k: v for k, v in agg_dict.items() if k in available_cols}
        
        resampled = df.resample(self.target_frequency).agg(agg_dict)
        
        return resampled

    def process_pv_data(self, filepath: str) -> pd.DataFrame:
        """Process PV production data."""
        print("\nReading PV data file...")
        
        # First read just the header to get column names
        header_df = pd.read_csv(filepath, sep=';', nrows=0)
        print("Column names in file:", header_df.columns.tolist())
        
        # Read the data, skipping the units row
        df = pd.read_csv(filepath, sep=';', skiprows=1, decimal=',', 
                        thousands=None, na_values=['n/a', '#WERT!'],
                        encoding='utf-8')
        
        print("First few rows of raw data:")
        print(df.head())
        print("\nColumns after reading:", df.columns.tolist())
        
        # Drop the units row (contains [dd.MM.yyyy HH:mm], [Wh], etc.)
        df = df[~df.iloc[:, 0].str.contains(r'\[.*\]', na=False)]
        
        # Strip any whitespace from values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()
        
        # Convert datetime
        date_col = df.columns[0]  # First column should be the date
        print(f"\nUsing {date_col} as date column")
        df[date_col] = pd.to_datetime(df[date_col], format='%d.%m.%Y %H:%M')
        
        # Set datetime as index
        df.set_index(date_col, inplace=True)
        
        # Rename columns for consistency
        df.columns = ['energy_wh', 'energy_interval', 'power_w']
        
        # Convert to numeric, coercing errors to NaN
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print("\nProcessed data shape:", df.shape)
        print("Sample of processed data:")
        print(df.head())
        
        # Resample to target frequency
        resampled = df.resample(self.target_frequency).agg({
            'energy_wh': 'last',  # Cumulative value
            'energy_interval': 'sum',  # Sum up interval energy
            'power_w': 'mean'  # Average power
        })
        
        return resampled

    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived time-based features."""
        # Add time-based features
        df['hour'] = df.index.hour
        df['day_of_year'] = df.index.dayofyear
        
        # Add solar position features (simplified)
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
        df['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)
        
        return df

    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize different types of features appropriately."""
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        import joblib
        import os

        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)

        # Initialize scalers
        minmax_scaler = MinMaxScaler()
        standard_scaler = StandardScaler()

        # Group columns by normalization type
        minmax_columns = [
            # INCA Radiation features
            'INCA_GlobalRadiation [W m-2]',
            'INCA_ClearSkyGHI',
            'INCA_ClearSkyDNI',
            'INCA_ClearSkyDHI',
            # Station Radiation features
            'Station_GlobalRadiation [W m-2]',
            'Station_ClearSkyGHI',
            # Combined Radiation features
            'Combined_GlobalRadiation [W m-2]',
            # PV output features
            'power_w', 'energy_wh', 'energy_interval',
            # Cyclic features already normalized
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
        ]
        
        standard_columns = [
            # INCA weather measurements
            'INCA_Temperature [degree_Celsius]',
            'INCA_RelativeHumidity [percent]',
            'INCA_WindSpeed [m s-1]',
            'INCA_WindU [m s-1]',
            'INCA_WindV [m s-1]',
            'INCA_ClearSkyIndex',
            # Station weather measurements
            'Station_Temperature [degree_Celsius]',
            'Station_WindSpeed [m s-1]',
            'Station_Precipitation [mm]',
            'Station_Pressure [hPa]',
            'Station_ClearSkyIndex',
            # Combined measurements
            'Combined_Temperature [degree_Celsius]',
            'Combined_WindSpeed [m s-1]',
            'Combined_ClearSkyIndex'
        ]

        # Filter columns that actually exist in the dataframe
        minmax_columns = [col for col in minmax_columns if col in df.columns]
        standard_columns = [col for col in standard_columns if col in df.columns]

        # Apply MinMaxScaler to radiation and PV output features (0 to 1 range)
        if minmax_columns:
            df[minmax_columns] = minmax_scaler.fit_transform(df[minmax_columns])
            joblib.dump(minmax_scaler, 'models/minmax_scaler.pkl')

        # Apply StandardScaler to weather measurements (zero mean, unit variance)
        if standard_columns:
            df[standard_columns] = standard_scaler.fit_transform(df[standard_columns])
            joblib.dump(standard_scaler, 'models/standard_scaler.pkl')

        # Hour and day of year are kept as is since they're already normalized cyclically
        return df

    def merge_datasets(self, inca_df: pd.DataFrame, station_df: pd.DataFrame,
                        pv_df: pd.DataFrame) -> pd.DataFrame:
        """Merge all datasets on timestamp index and combine overlapping features."""
        print("\nDataset shapes before merge:")
        print(f"INCA data: {inca_df.shape}")
        print(f"Station data: {station_df.shape}")
        print(f"PV data: {pv_df.shape}")
        
        # Merge dataframes
        merged = pd.concat([inca_df, station_df, pv_df], axis=1)
        print(f"\nMerged shape before dropna: {merged.shape}")
        
        # Remove rows with missing values
        merged = merged.dropna()
        print(f"Merged shape after dropna: {merged.shape}")
        
        # Calculate station clear sky index if station radiation data exists
        if 'Station_GlobalRadiation [W m-2]' in merged.columns:
            # Calculate clear sky values for station data
            station_clear_sky = self.calculate_clear_sky(merged)
            merged['Station_ClearSkyGHI'] = station_clear_sky['INCA_ClearSkyGHI']
            
            # Calculate station clear sky index
            merged['Station_ClearSkyIndex'] = np.where(
                merged['Station_ClearSkyGHI'] > 10,
                merged['Station_GlobalRadiation [W m-2]'] / merged['Station_ClearSkyGHI'],
                0
            )
            merged['Station_ClearSkyIndex'] = merged['Station_ClearSkyIndex'].clip(0, 1.5)
        
        # Combine overlapping features
        # 1. Global Radiation
        if all(col in merged.columns for col in ['INCA_GlobalRadiation [W m-2]', 'Station_GlobalRadiation [W m-2]']):
            merged['Combined_GlobalRadiation [W m-2]'] = merged[['INCA_GlobalRadiation [W m-2]', 'Station_GlobalRadiation [W m-2]']].mean(axis=1)
            
        # 2. Temperature
        if all(col in merged.columns for col in ['INCA_Temperature [degree_Celsius]', 'Station_Temperature [degree_Celsius]']):
            merged['Combined_Temperature [degree_Celsius]'] = merged[['INCA_Temperature [degree_Celsius]', 'Station_Temperature [degree_Celsius]']].mean(axis=1)
            
        # 3. Wind Speed
        if all(col in merged.columns for col in ['INCA_WindSpeed [m s-1]', 'Station_WindSpeed [m s-1]']):
            merged['Combined_WindSpeed [m s-1]'] = merged[['INCA_WindSpeed [m s-1]', 'Station_WindSpeed [m s-1]']].mean(axis=1)
            
        # 4. Clear Sky Index
        if all(col in merged.columns for col in ['INCA_ClearSkyIndex', 'Station_ClearSkyIndex']):
            merged['Combined_ClearSkyIndex'] = merged[['INCA_ClearSkyIndex', 'Station_ClearSkyIndex']].mean(axis=1)
        
        # Add derived features
        merged = self.add_derived_features(merged)
        
        # Normalize the data
        print("\nNormalizing data...")
        merged = self.normalize_data(merged)
        print("Data normalization complete")
        
        return merged

    def save_to_parquet(self, df: pd.DataFrame, output_path: str):
        """Save processed data to parquet format with compression."""
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_path, compression='snappy')

    def calculate_clear_sky(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate clear sky irradiance values using pvlib."""
        # Define location parameters
        latitude = 47.38770748541585
        longitude = 15.094127778561258
        altitude = 541  # approximate altitude for Leoben
        tz = 'Etc/GMT+1'
        location = Location(latitude, longitude, tz, altitude)

        # Get solar position
        solar_position = location.get_solarposition(df.index)
        
        # Calculate pressure based on altitude (approximately) and convert from hPa to Pa
        pressure = pvlib.atmosphere.alt2pres(altitude) * 100  # Convert from hPa to Pa
        
        # Get precipitable water from relative humidity and temperature
        # Using a simple approximation based on temperature and RH
        temp_air = df['INCA_Temperature [degree_Celsius]'].values
        relative_humidity = df['INCA_RelativeHumidity [percent]'].values
        # Approximate precipitable water (in cm) using a simple model
        # This is a rough approximation - could be improved with more sophisticated models
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

        # Add clear sky values to dataframe with INCA prefix
        df['INCA_ClearSkyGHI'] = clear_sky['ghi']
        df['INCA_ClearSkyDNI'] = clear_sky['dni']
        df['INCA_ClearSkyDHI'] = clear_sky['dhi']

        # Calculate clear sky index (ratio of measured to clear sky GHI)
        # Handle division by zero and very small values
        df['INCA_ClearSkyIndex'] = np.where(
            df['INCA_ClearSkyGHI'] > 10,  # Only calculate for significant irradiance
            df['INCA_GlobalRadiation [W m-2]'] / df['INCA_ClearSkyGHI'],
            0  # Set to 0 for nighttime or very low irradiance
        )

        # Clip unrealistic clear sky index values
        df['INCA_ClearSkyIndex'] = df['INCA_ClearSkyIndex'].clip(0, 1.5)  # Typical range is 0-1.2

        return df

    def process_all(self, inca_path: str, station_path: str, pv_path: str,
                   output_path: str):
        """Process all datasets and save the result."""
        print("Processing INCA data...")
        inca_df = self.process_inca_data(inca_path)
        print(f"INCA data shape: {inca_df.shape}")
        
        print("\nProcessing station data...")
        station_df = self.process_station_data(station_path)
        print(f"Station data shape: {station_df.shape}")
        
        print("\nProcessing PV data...")
        pv_df = self.process_pv_data(pv_path)
        print(f"PV data shape: {pv_df.shape}")
        
        print("\nMerging datasets...")
        merged_df = self.merge_datasets(inca_df, station_df, pv_df)
        
        print("\nSaving to parquet...")
        self.save_to_parquet(merged_df, output_path)
        
        print(f"\nProcessing complete. Output saved to {output_path}")
        print(f"Final dataset shape: {merged_df.shape}")
        
        # Print sample of final dataset
        print("\nSample of processed data:")
        print(merged_df.head())
        
        return merged_df

if __name__ == "__main__":
    # File paths
    INCA_PATH = "data/inca_data_evt/INCA analysis - large domain Datensatz_20220713T0000_20250304T2300 (1).csv"
    STATION_PATH = "data/station_data_leoben/Messstationen Zehnminutendaten v2 Datensatz_20220713T0000_20250304T0000.csv"
    PV_PATH = "data/raw_data_evt_act/merge.CSV"
    OUTPUT_PATH = "data/processed_training_data.parquet"
    
    # Initialize and run processor
    processor = DataProcessor(chunk_size=100000)
    df = processor.process_all(INCA_PATH, STATION_PATH, PV_PATH, OUTPUT_PATH)