#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Timestamp Mismatch Analysis and Correction

This script analyzes potential timestamp mismatches between the weather station data
and PV power output data, and creates corrected datasets with aligned timestamps.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import signal
from scipy.stats import pearsonr
import os
from datetime import timedelta

def analyze_timestamp_mismatch():
    """Analyze and correct timestamp mismatches between station and PV data."""
    print("Loading data files...")
    df_10min = pd.read_parquet('data/station_data_10min.parquet')
    df_1h = pd.read_parquet('data/station_data_1h.parquet')

    print(f'10-minute data: {df_10min.shape} rows, date range: {df_10min.index.min()} to {df_10min.index.max()}')
    print(f'1-hour data: {df_1h.shape} rows, date range: {df_1h.index.min()} to {df_1h.index.max()}')

    # Create output directory
    os.makedirs('data/corrected', exist_ok=True)
    os.makedirs('visualizations/timestamp_analysis', exist_ok=True)

    # 1. Find the first date with non-zero PV power
    print("\nAnalyzing PV data start date...")
    
    # For 10-minute data
    pv_start_10min = df_10min[df_10min['power_w'] > 0].index.min()
    print(f"First non-zero PV power in 10-minute data: {pv_start_10min}")
    
    # For 1-hour data
    pv_start_1h = df_1h[df_1h['power_w'] > 0].index.min()
    print(f"First non-zero PV power in 1-hour data: {pv_start_1h}")
    
    # 2. Cross-correlation analysis to detect lag
    print("\nPerforming cross-correlation analysis to detect lag...")
    
    # Function to find optimal lag using cross-correlation
    def find_optimal_lag(radiation, power, max_lag_hours=3):
        # Convert max_lag_hours to number of samples
        max_lag = max_lag_hours * (60 // pd.Timedelta(power.index[1] - power.index[0]).seconds * 60)
        
        # Compute cross-correlation
        correlation = signal.correlate(power, radiation, mode='full')
        lags = signal.correlation_lags(len(power), len(radiation), mode='full')
        
        # Find lag with maximum correlation within the specified range
        valid_lags = (lags >= -max_lag) & (lags <= max_lag)
        valid_correlation = correlation[valid_lags]
        valid_lags = lags[valid_lags]
        
        if len(valid_lags) > 0:
            optimal_lag = valid_lags[np.argmax(valid_correlation)]
            max_corr = np.max(valid_correlation)
            return optimal_lag, max_corr
        else:
            return 0, 0
    
    # Analyze lag for a sample period (e.g., one clear day)
    # Find a clear day with good radiation and power data
    clear_days = ['2022-08-03', '2022-08-04', '2022-08-05']  # Example clear days based on the plot
    
    for day in clear_days:
        print(f"\nAnalyzing lag for {day}...")
        
        # Extract data for the day
        day_data_1h = df_1h.loc[day]
        
        # Normalize data for better correlation analysis
        radiation = (day_data_1h['GlobalRadiation [W m-2]'] - day_data_1h['GlobalRadiation [W m-2]'].min()) / \
                    (day_data_1h['GlobalRadiation [W m-2]'].max() - day_data_1h['GlobalRadiation [W m-2]'].min())
        power = (day_data_1h['power_w'] - day_data_1h['power_w'].min()) / \
                (day_data_1h['power_w'].max() - day_data_1h['power_w'].min())
        
        # Find optimal lag
        optimal_lag, max_corr = find_optimal_lag(radiation.values, power.values)
        print(f"Optimal lag for {day}: {optimal_lag} samples, correlation: {max_corr:.4f}")
        
        # Calculate lag in minutes for 1-hour data
        lag_minutes = optimal_lag * 60  # Each sample is 1 hour
        print(f"Estimated lag: {lag_minutes} minutes")
        
        # Plot original and shifted data
        plt.figure(figsize=(12, 6))
        plt.plot(day_data_1h.index, radiation, 'b-', label='Global Radiation (normalized)')
        plt.plot(day_data_1h.index, power, 'r-', label='Power Output (normalized)')
        
        # If there's a significant lag, plot shifted power
        if abs(optimal_lag) >= 1:
            # Shift the power data
            shifted_power = np.roll(power.values, -optimal_lag)
            plt.plot(day_data_1h.index, shifted_power, 'g--', label=f'Power Output (shifted by {lag_minutes} min)')
        
        plt.title(f'Radiation and Power Output Comparison - {day}')
        plt.xlabel('Time')
        plt.ylabel('Normalized Value')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'visualizations/timestamp_analysis/lag_analysis_{day}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Systematic lag analysis across multiple days
    print("\nPerforming systematic lag analysis across multiple days...")
    
    # Analyze lag for each day in a month
    start_date = '2022-08-01'
    end_date = '2022-08-31'
    
    daily_lags = []
    daily_corrs = []
    dates = []
    
    for day in pd.date_range(start=start_date, end=end_date, freq='D'):
        day_str = day.strftime('%Y-%m-%d')
        
        try:
            # Extract data for the day
            day_data_1h = df_1h.loc[day_str]
            
            # Skip days with insufficient data
            if len(day_data_1h) < 12 or day_data_1h['power_w'].max() < 100:
                continue
            
            # Normalize data
            radiation = (day_data_1h['GlobalRadiation [W m-2]'] - day_data_1h['GlobalRadiation [W m-2]'].min()) / \
                        (day_data_1h['GlobalRadiation [W m-2]'].max() - day_data_1h['GlobalRadiation [W m-2]'].min() + 1e-10)
            power = (day_data_1h['power_w'] - day_data_1h['power_w'].min()) / \
                    (day_data_1h['power_w'].max() - day_data_1h['power_w'].min() + 1e-10)
            
            # Find optimal lag
            optimal_lag, max_corr = find_optimal_lag(radiation.values, power.values)
            
            daily_lags.append(optimal_lag)
            daily_corrs.append(max_corr)
            dates.append(day)
            
            print(f"Day: {day_str}, Lag: {optimal_lag} samples, Correlation: {max_corr:.4f}")
            
        except Exception as e:
            print(f"Error analyzing {day_str}: {e}")
    
    # Calculate the most common lag
    if daily_lags:
        most_common_lag = int(round(np.median(daily_lags)))
        avg_correlation = np.mean(daily_corrs)
        print(f"\nMost common lag: {most_common_lag} samples (median)")
        print(f"Average correlation: {avg_correlation:.4f}")
        
        # Plot the distribution of lags
        plt.figure(figsize=(10, 6))
        plt.hist(daily_lags, bins=range(min(daily_lags)-1, max(daily_lags)+2), alpha=0.7)
        plt.axvline(most_common_lag, color='r', linestyle='--', label=f'Median Lag: {most_common_lag}')
        plt.title('Distribution of Timestamp Lags Between Radiation and Power')
        plt.xlabel('Lag (samples)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.savefig('visualizations/timestamp_analysis/lag_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot lags over time
        plt.figure(figsize=(12, 6))
        plt.plot(dates, daily_lags, 'o-')
        plt.axhline(most_common_lag, color='r', linestyle='--', label=f'Median Lag: {most_common_lag}')
        plt.title('Timestamp Lag Between Radiation and Power Over Time')
        plt.xlabel('Date')
        plt.ylabel('Lag (samples)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('visualizations/timestamp_analysis/lag_over_time.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Create corrected datasets
        print("\nCreating corrected datasets...")
        
        # Calculate lag in minutes
        lag_minutes_1h = most_common_lag * 60  # For 1-hour data
        lag_minutes_10min = most_common_lag * 6  # For 10-minute data (assuming 10 samples per hour)
        
        # Create corrected datasets by shifting the timestamps
        if most_common_lag != 0:
            # For 1-hour data
            df_1h_corrected = df_1h.copy()
            df_1h_corrected.index = df_1h_corrected.index - timedelta(minutes=lag_minutes_1h)
            
            # For 10-minute data
            df_10min_corrected = df_10min.copy()
            df_10min_corrected.index = df_10min_corrected.index - timedelta(minutes=lag_minutes_10min)
            
            # Save corrected datasets
            df_1h_corrected.to_parquet('data/corrected/station_data_1h_corrected.parquet')
            df_10min_corrected.to_parquet('data/corrected/station_data_10min_corrected.parquet')
            
            print(f"Corrected datasets saved with {lag_minutes_1h} minutes lag adjustment for 1-hour data")
            print(f"and {lag_minutes_10min} minutes lag adjustment for 10-minute data")
            
            # 5. Visualize original vs corrected data
            print("\nVisualizing original vs corrected data...")
            
            # Sample one week of data
            start_date = '2022-08-01'
            end_date = '2022-08-07'
            
            # For 1-hour data
            df_week_1h = df_1h.loc[start_date:end_date]
            df_week_1h_corrected = df_1h_corrected.loc[start_date:end_date]
            
            # Plot original data
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            color = 'tab:blue'
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Global Radiation [W/m²]', color=color)
            ax1.plot(df_week_1h.index, df_week_1h['GlobalRadiation [W m-2]'], color=color, label='Global Radiation')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax1.xaxis.set_major_locator(mdates.DayLocator())
            
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Power [W]', color=color)
            ax2.plot(df_week_1h.index, df_week_1h['power_w'], color=color, label='Power Output')
            ax2.tick_params(axis='y', labelcolor=color)
            
            # Add combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            plt.title('Original Data: Solar Radiation and PV Power Output (Hourly)')
            plt.tight_layout()
            plt.savefig('visualizations/timestamp_analysis/original_data.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot corrected data
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            color = 'tab:blue'
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Global Radiation [W/m²]', color=color)
            ax1.plot(df_week_1h.index, df_week_1h['GlobalRadiation [W m-2]'], color=color, label='Global Radiation')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax1.xaxis.set_major_locator(mdates.DayLocator())
            
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Power [W]', color=color)
            ax2.plot(df_week_1h_corrected.index, df_week_1h_corrected['power_w'], color=color, label='Power Output (Corrected)')
            ax2.tick_params(axis='y', labelcolor=color)
            
            # Add combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            plt.title(f'Corrected Data: Solar Radiation and PV Power Output (Hourly, {lag_minutes_1h}min shift)')
            plt.tight_layout()
            plt.savefig('visualizations/timestamp_analysis/corrected_data.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 6. Trim data to start from first PV power
            print("\nTrimming data to start from first PV power...")
            
            # Find the first date with non-zero PV power in corrected data
            pv_start_1h_corrected = df_1h_corrected[df_1h_corrected['power_w'] > 0].index.min()
            pv_start_10min_corrected = df_10min_corrected[df_10min_corrected['power_w'] > 0].index.min()
            
            print(f"First non-zero PV power in corrected 1-hour data: {pv_start_1h_corrected}")
            print(f"First non-zero PV power in corrected 10-minute data: {pv_start_10min_corrected}")
            
            # Trim data to start from first PV power
            df_1h_trimmed = df_1h_corrected.loc[pv_start_1h_corrected:]
            df_10min_trimmed = df_10min_corrected.loc[pv_start_10min_corrected:]
            
            # Save trimmed datasets
            df_1h_trimmed.to_parquet('data/corrected/station_data_1h_trimmed.parquet')
            df_10min_trimmed.to_parquet('data/corrected/station_data_10min_trimmed.parquet')
            
            print(f"Trimmed datasets saved, starting from {pv_start_1h_corrected} for 1-hour data")
            print(f"and {pv_start_10min_corrected} for 10-minute data")
            
            # Visualize trimmed data
            # Sample one week of data from the start of PV power
            start_date = pv_start_1h_corrected
            end_date = start_date + timedelta(days=7)
            
            df_week_1h_trimmed = df_1h_trimmed.loc[start_date:end_date]
            
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            color = 'tab:blue'
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Global Radiation [W/m²]', color=color)
            ax1.plot(df_week_1h_trimmed.index, df_week_1h_trimmed['GlobalRadiation [W m-2]'], color=color, label='Global Radiation')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax1.xaxis.set_major_locator(mdates.DayLocator())
            
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Power [W]', color=color)
            ax2.plot(df_week_1h_trimmed.index, df_week_1h_trimmed['power_w'], color=color, label='Power Output')
            ax2.tick_params(axis='y', labelcolor=color)
            
            # Add combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            plt.title('Trimmed Data: Solar Radiation and PV Power Output (Hourly)')
            plt.tight_layout()
            plt.savefig('visualizations/timestamp_analysis/trimmed_data.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            print("No significant lag detected. No correction needed.")
    else:
        print("Insufficient data for lag analysis.")

if __name__ == "__main__":
    analyze_timestamp_mismatch()