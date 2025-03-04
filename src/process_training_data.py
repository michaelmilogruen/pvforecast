import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import Dict, List, Optional
import pyarrow as pa
import pyarrow.parquet as pq

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
        
        # Select relevant columns
        columns = ['GL [W m-2]', 'T2M [degree_Celsius]', 'RH2M [percent]', 
                  'UU [m s-1]', 'VV [m s-1]']
        df = df[columns]
        
        # Resample to target frequency using appropriate methods
        resampled = df.resample(self.target_frequency).interpolate(method='linear')
        
        return resampled

    def process_station_data(self, filepath: str) -> pd.DataFrame:
        """Process local weather station data (10min) with quality filtering."""
        df = self.read_csv_in_chunks(
            filepath=filepath,
            date_column='time',
            date_format='%Y-%m-%dT%H:%M+00:00'
        )
        
        # Define critical measurements and their quality flags
        critical_columns = {
            'cglo': 'cglo_flag',
            'tl': 'tl_flag',
            'ff': 'ff_flag'
        }
        
        # Filter based on quality flags (flag value 12 seems to be good based on data inspection)
        for col, flag_col in critical_columns.items():
            if col in df.columns and flag_col in df.columns:
                df = df[df[flag_col] == 12]
        
        # Select relevant columns (excluding flag columns)
        relevant_cols = ['cglo', 'tl', 'ff', 'rr', 'p']
        available_cols = [col for col in relevant_cols if col in df.columns]
        df = df[available_cols]
        
        # Resample to target frequency
        agg_dict = {
            'cglo': 'mean',
            'tl': 'mean',
            'ff': 'mean',
            'rr': 'sum',
            'p': 'mean'
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

    def merge_datasets(self, inca_df: pd.DataFrame, station_df: pd.DataFrame, 
                      pv_df: pd.DataFrame) -> pd.DataFrame:
        """Merge all datasets on timestamp index."""
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
        
        # Add derived features
        merged = self.add_derived_features(merged)
        
        return merged

    def save_to_parquet(self, df: pd.DataFrame, output_path: str):
        """Save processed data to parquet format with compression."""
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_path, compression='snappy')

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