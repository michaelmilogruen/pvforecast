#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to merge all PV data files from data/raw_data_evt_act/ directory into a single CSV file.
This script handles both daily and 5-minute interval data formats, and both CSV and Excel files.

Author: Michael Grün
Date: 2025-03-03
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import glob

def detect_file_format(df):
    """
    Detect if the file contains daily data or 5-minute interval data.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the data
    
    Returns:
        str: 'daily' or '5min' indicating the data format
    """
    # Check column names
    columns = df.columns.tolist()
    
    # First check if we have the expected columns
    if 'Datum und Uhrzeit' in columns:
        # Check if the first value contains time
        if len(df) > 0:
            first_val = str(df['Datum und Uhrzeit'].iloc[0])
            if ':' in first_val:  # Contains time (HH:MM)
                return '5min'
    
    # Check for other column patterns that might indicate 5-minute data
    time_related_cols = [col for col in columns if any(term in col.lower() for term in ['zeit', 'time', 'datum', 'date'])]
    if time_related_cols:
        for col in time_related_cols:
            if len(df) > 0:
                first_val = str(df[col].iloc[0])
                if ':' in first_val:  # Contains time (HH:MM)
                    return '5min'
    
    # Otherwise assume it's daily data
    return 'daily'

def read_file(file_path):
    """
    Read a data file (CSV or Excel) and return a DataFrame.
    
    Args:
        file_path (str): Path to the file
    
    Returns:
        tuple: (pandas.DataFrame, str) - DataFrame and detected format ('daily' or '5min')
    """
    try:
        if file_path.endswith('.csv'):
            # Read CSV file
            df = pd.read_csv(file_path, encoding='utf-8')
        elif file_path.endswith('.xlsx'):
            # Read Excel file, skip the first row which often contains format info
            try:
                # First try with engine='openpyxl' which is more stable
                df = pd.read_excel(file_path, engine='openpyxl', skiprows=1)
            except Exception as excel_error:
                print(f"Error reading Excel file with openpyxl: {excel_error}")
                try:
                    # Try with default engine
                    df = pd.read_excel(file_path, skiprows=1)
                except Exception as excel_error2:
                    print(f"Error reading Excel file: {excel_error2}")
                    return None, None
        else:
            print(f"Unsupported file format: {file_path}")
            return None, None
        
        # Detect format
        file_format = detect_file_format(df)
        
        return df, file_format
    
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None, None

def process_daily_data(df, file_name):
    """
    Process daily data format.
    
    Args:
        df (pandas.DataFrame): DataFrame containing daily data
        file_name (str): Original file name for reference
    
    Returns:
        pandas.DataFrame: Processed DataFrame with standardized columns
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Identify date column
    date_col = None
    if 'Datum und Uhrzeit' in df.columns:
        date_col = 'Datum und Uhrzeit'
    else:
        # Look for columns that might contain date information
        for col in df.columns:
            if any(term in col.lower() for term in ['datum', 'date', 'zeit', 'time']):
                date_col = col
                break
    
    if not date_col:
        print(f"Could not find date column in {file_name}")
        return None
    
    # Standardize column names
    df.rename(columns={date_col: 'Date'}, inplace=True)
    
    # Ensure date is in datetime format - use pd.to_datetime for consistency
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
    except:
        try:
            # Try another common format
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        except Exception as e:
            print(f"Error converting dates in {file_name}: {e}")
            return None
    
    # Find energy production column - could be named differently in different files
    energy_col = None
    for col in df.columns:
        if any(term in col for term in ['Energie', 'Energy', 'Gesamtanlage', 'PV Produktion', 'Symo']):
            energy_col = col
            break
    
    if not energy_col:
        # If we couldn't find a specific column, use the second column (common in these files)
        if len(df.columns) >= 2:
            energy_col = df.columns[1]
        else:
            print(f"Could not find energy column in {file_name}")
            return None
    
    # Convert energy values to numeric
    df[energy_col] = pd.to_numeric(df[energy_col], errors='coerce')
    
    # Create standardized DataFrame
    result_df = pd.DataFrame({
        'Date': df['Date'],
        'Energy_kWh': df[energy_col],
        'Source_File': file_name,
        'Data_Type': 'daily'
    })
    return result_df

def process_5min_data(df, file_name):
    """
    Process 5-minute interval data format.
    
    Args:
        df (pandas.DataFrame): DataFrame containing 5-minute interval data
        file_name (str): Original file name for reference
    
    Returns:
        pandas.DataFrame: Processed DataFrame with standardized columns
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Identify timestamp column
    timestamp_col = None
    if 'Datum und Uhrzeit' in df.columns:
        timestamp_col = 'Datum und Uhrzeit'
    else:
        # Look for columns that might contain timestamp information
        for col in df.columns:
            if any(term in col.lower() for term in ['zeit', 'time', 'datum', 'date']):
                if len(df) > 0:
                    first_val = str(df[col].iloc[0])
                    if ':' in first_val:  # Contains time (HH:MM)
                        timestamp_col = col
                        break
    
    if not timestamp_col:
        print(f"Could not find timestamp column in {file_name}")
        return None, None
    
    # Standardize column names
    df.rename(columns={timestamp_col: 'Timestamp'}, inplace=True)
    
    # Ensure timestamp is in datetime format
    try:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d.%m.%Y %H:%M', errors='coerce')
    except:
        try:
            # Try another common format
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        except Exception as e:
            print(f"Error converting timestamps in {file_name}: {e}")
            return None, None
    
    # Extract date for grouping - convert to pandas Timestamp for consistency
    df['Date'] = pd.to_datetime(df['Timestamp'].dt.date)
    
    # Find power/energy production column - could be named differently in different files
    power_col = None
    for col in df.columns:
        if any(term in col for term in ['PV Produktion', 'Energie', 'Energy', 'Symo']):
            power_col = col
            break
    
    if not power_col:
        # If we couldn't find a specific column, use the second column (common in these files)
        if len(df.columns) >= 2:
            power_col = df.columns[1]
        else:
            print(f"Could not find power/energy column in {file_name}")
            return None, None
    
    # Replace 'n/a' with NaN
    df[power_col] = df[power_col].replace('n/a', np.nan)
    
    # Convert to numeric
    df[power_col] = pd.to_numeric(df[power_col], errors='coerce')
    
    # Create standardized DataFrame
    result_df = pd.DataFrame({
        'Timestamp': df['Timestamp'],
        'Date': df['Date'],
        'Power_Wh': df[power_col],
        'Source_File': file_name,
        'Data_Type': '5min'
    })
    
    # Also create a daily aggregation
    daily_df = result_df.groupby('Date')['Power_Wh'].sum().reset_index()
    daily_df['Energy_kWh'] = daily_df['Power_Wh'] / 1000  # Convert Wh to kWh
    daily_df['Source_File'] = file_name
    daily_df['Data_Type'] = '5min_aggregated'
    daily_df = daily_df[['Date', 'Energy_kWh', 'Source_File', 'Data_Type']]
    
    return result_df, daily_df

def merge_data_files():
    """
    Merge all data files from data/raw_data_evt_act/ directory.
    
    Returns:
        tuple: (pandas.DataFrame, pandas.DataFrame) - 5-minute data and daily data
    """
    data_dir = "data/raw_data_evt_act"
    
    # Get all CSV and Excel files
    file_paths = glob.glob(os.path.join(data_dir, "*.csv"))
    file_paths.extend(glob.glob(os.path.join(data_dir, "*.xlsx")))
    
    # Initialize empty DataFrames for different data types
    all_5min_data = []
    all_daily_data = []
    
    print(f"Found {len(file_paths)} files to process")
    
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        print(f"Processing {file_name}...")
        
        df, file_format = read_file(file_path)
        
        if df is not None:
            if file_format == 'daily':
                processed_df = process_daily_data(df, file_name)
                if processed_df is not None:
                    all_daily_data.append(processed_df)
            elif file_format == '5min':
                processed_5min_df, processed_daily_df = process_5min_data(df, file_name)
                if processed_5min_df is not None:
                    all_5min_data.append(processed_5min_df)
                if processed_daily_df is not None:
                    all_daily_data.append(processed_daily_df)
    
    # Combine all data
    combined_5min_data = pd.DataFrame()
    combined_daily_data = pd.DataFrame()
    
    if all_5min_data:
        combined_5min_data = pd.concat(all_5min_data, ignore_index=True)
        # Sort by timestamp
        if not combined_5min_data.empty:
            combined_5min_data.sort_values('Timestamp', inplace=True)
    
    if all_daily_data:
        combined_daily_data = pd.concat(all_daily_data, ignore_index=True)
        # Sort by date
        if not combined_daily_data.empty:
            combined_daily_data.sort_values('Date', inplace=True)
            # Remove duplicates (keep the first occurrence)
            combined_daily_data.drop_duplicates(subset=['Date'], keep='first', inplace=True)
    
    return combined_5min_data, combined_daily_data

def main():
    """
    Main function to merge data files and save results.
    """
    print("Starting data merge process...")
    
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Merge data files
    combined_5min_data, combined_daily_data = merge_data_files()
    
    # Save 5-minute data
    if not combined_5min_data.empty:
        output_5min_file = f"data/merged_5min_data_{timestamp}.csv"
        combined_5min_data.to_csv(output_5min_file, index=False)
        print(f"Saved 5-minute data to {output_5min_file}")
        print(f"Total rows in 5-minute data: {len(combined_5min_data)}")
    else:
        print("No 5-minute data found")
    
    # Save daily data
    if not combined_daily_data.empty:
        output_daily_file = f"data/merged_daily_data_{timestamp}.csv"
        combined_daily_data.to_csv(output_daily_file, index=False)
        print(f"Saved daily data to {output_daily_file}")
        print(f"Total rows in daily data: {len(combined_daily_data)}")
    else:
        print("No daily data found")
    
    # Create a single merged file with all data
    output_all_file = f"data/merged_all_data_{timestamp}.csv"
    
    # If we have both types of data, create a combined file
    if not combined_daily_data.empty:
        # Save the merged file
        combined_daily_data.to_csv(output_all_file, index=False)
        print(f"Saved all merged data to {output_all_file}")
        print(f"Total rows in merged data: {len(combined_daily_data)}")
    
    print("Data merge process completed!")

if __name__ == "__main__":
    main()