#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Temporal Resolution Analysis for PV Forecasting

This script analyzes the impact of different temporal resolutions (10-minute vs 1-hour)
on the characteristics of the data and their implications for deep learning models.
It creates visualizations suitable for scientific papers.

Key analyses:
1. Information loss in downsampling
2. Feature distribution changes across resolutions
3. Autocorrelation differences
4. Spectral analysis
5. Implications for deep learning model design
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import signal, stats
from statsmodels.graphics.tsaplots import plot_acf
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

def create_resolution_comparison_visualizations():
    """Create visualizations comparing different temporal resolutions."""
    # Create visualizations directory if it doesn't exist
    os.makedirs('visualizations/resolution_comparison', exist_ok=True)
    
    # Set plot style for scientific publication
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 20

    # Load data
    print("Loading data files...")
    df_10min = pd.read_parquet('data/station_data_10min.parquet')
    df_1h = pd.read_parquet('data/station_data_1h.parquet')

    print(f'10-minute data: {df_10min.shape} rows, date range: {df_10min.index.min()} to {df_10min.index.max()}')
    print(f'1-hour data: {df_1h.shape} rows, date range: {df_1h.index.min()} to {df_1h.index.max()}')

    # 1. Information Loss in Downsampling
    print("Analyzing information loss in downsampling...")
    
    # Select a sample day for detailed comparison
    sample_day = '2022-08-01'
    df_day_10min = df_10min.loc[sample_day]
    df_day_1h = df_1h.loc[sample_day]
    
    # Create a resampled version from 10min to 1h for direct comparison
    df_10min_resampled = df_10min.loc[sample_day].resample('1H').mean()
    
    # Plot comparison of original 10min, resampled 1h, and original 1h data
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Global Radiation
    axes[0].plot(df_day_10min.index, df_day_10min['GlobalRadiation [W m-2]'], 'b-', alpha=0.5, label='10-minute Resolution')
    axes[0].plot(df_day_1h.index, df_day_1h['GlobalRadiation [W m-2]'], 'r-o', alpha=0.8, label='1-hour Resolution')
    axes[0].plot(df_10min_resampled.index, df_10min_resampled['GlobalRadiation [W m-2]'], 'g--s', alpha=0.8, label='10-min Resampled to 1-hour')
    axes[0].set_title('Global Radiation: Resolution Comparison')
    axes[0].set_ylabel('Global Radiation [W/m²]')
    axes[0].legend()
    
    # Temperature
    axes[1].plot(df_day_10min.index, df_day_10min['Temperature [degree_Celsius]'], 'b-', alpha=0.5, label='10-minute Resolution')
    axes[1].plot(df_day_1h.index, df_day_1h['Temperature [degree_Celsius]'], 'r-o', alpha=0.8, label='1-hour Resolution')
    axes[1].plot(df_10min_resampled.index, df_10min_resampled['Temperature [degree_Celsius]'], 'g--s', alpha=0.8, label='10-min Resampled to 1-hour')
    axes[1].set_title('Temperature: Resolution Comparison')
    axes[1].set_ylabel('Temperature [°C]')
    axes[1].legend()
    
    # Power Output
    axes[2].plot(df_day_10min.index, df_day_10min['power_w'], 'b-', alpha=0.5, label='10-minute Resolution')
    axes[2].plot(df_day_1h.index, df_day_1h['power_w'], 'r-o', alpha=0.8, label='1-hour Resolution')
    axes[2].plot(df_10min_resampled.index, df_10min_resampled['power_w'], 'g--s', alpha=0.8, label='10-min Resampled to 1-hour')
    axes[2].set_title('Power Output: Resolution Comparison')
    axes[2].set_ylabel('Power [W]')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('visualizations/resolution_comparison/resolution_comparison_day.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Calculate information loss metrics
    print("Calculating information loss metrics...")
    
    # Calculate metrics for a longer period (e.g., one month)
    start_date = '2022-08-01'
    end_date = '2022-08-31'
    
    df_month_10min = df_10min.loc[start_date:end_date]
    df_month_1h = df_1h.loc[start_date:end_date]
    
    # Resample 10min data to 1h for comparison
    df_month_10min_resampled = df_month_10min.resample('1H').mean()
    
    # Align indices for comparison
    common_indices = df_month_1h.index.intersection(df_month_10min_resampled.index)
    df_1h_aligned = df_month_1h.loc[common_indices]
    df_10min_resampled_aligned = df_month_10min_resampled.loc[common_indices]
    
    # Calculate metrics for key variables
    variables = ['GlobalRadiation [W m-2]', 'Temperature [degree_Celsius]', 'power_w']
    metrics = {}
    
    for var in variables:
        # Calculate RMSE between original 1h and resampled 1h
        rmse = np.sqrt(mean_squared_error(df_1h_aligned[var], df_10min_resampled_aligned[var]))
        
        # Calculate MAE
        mae = mean_absolute_error(df_1h_aligned[var], df_10min_resampled_aligned[var])
        
        # Calculate correlation
        corr = np.corrcoef(df_1h_aligned[var], df_10min_resampled_aligned[var])[0, 1]
        
        metrics[var] = {'RMSE': rmse, 'MAE': mae, 'Correlation': corr}
    
    # Create a table of metrics
    metrics_df = pd.DataFrame(metrics).T
    
    # Plot metrics as a table
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=metrics_df.values.round(4),
                    rowLabels=metrics_df.index,
                    colLabels=metrics_df.columns,
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    plt.title('Information Loss Metrics: 1-hour vs Resampled 10-minute Data')
    plt.tight_layout()
    plt.savefig('visualizations/resolution_comparison/information_loss_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Feature Distribution Changes
    print("Analyzing feature distribution changes across resolutions...")
    
    # Compare distributions of key variables
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Global Radiation
    axes[0, 0].hist(df_10min['GlobalRadiation [W m-2]'].dropna(), bins=50, alpha=0.5, label='10-minute')
    axes[0, 0].hist(df_1h['GlobalRadiation [W m-2]'].dropna(), bins=50, alpha=0.5, label='1-hour')
    axes[0, 0].set_title('Global Radiation Distribution')
    axes[0, 0].set_xlabel('Global Radiation [W/m²]')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    # Q-Q plot for Global Radiation
    axes[0, 1].set_title('Q-Q Plot: Global Radiation')
    stats.probplot(df_10min['GlobalRadiation [W m-2]'].dropna(), dist="norm", plot=axes[0, 1])
    axes[0, 1].get_lines()[0].set_markerfacecolor('blue')
    axes[0, 1].get_lines()[0].set_label('10-minute')
    
    stats.probplot(df_1h['GlobalRadiation [W m-2]'].dropna(), dist="norm", plot=axes[0, 1])
    axes[0, 1].get_lines()[2].set_markerfacecolor('red')
    axes[0, 1].get_lines()[2].set_label('1-hour')
    axes[0, 1].legend()
    
    # Temperature
    axes[1, 0].hist(df_10min['Temperature [degree_Celsius]'].dropna(), bins=50, alpha=0.5, label='10-minute')
    axes[1, 0].hist(df_1h['Temperature [degree_Celsius]'].dropna(), bins=50, alpha=0.5, label='1-hour')
    axes[1, 0].set_title('Temperature Distribution')
    axes[1, 0].set_xlabel('Temperature [°C]')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # Q-Q plot for Temperature
    axes[1, 1].set_title('Q-Q Plot: Temperature')
    stats.probplot(df_10min['Temperature [degree_Celsius]'].dropna(), dist="norm", plot=axes[1, 1])
    axes[1, 1].get_lines()[0].set_markerfacecolor('blue')
    axes[1, 1].get_lines()[0].set_label('10-minute')
    
    stats.probplot(df_1h['Temperature [degree_Celsius]'].dropna(), dist="norm", plot=axes[1, 1])
    axes[1, 1].get_lines()[2].set_markerfacecolor('red')
    axes[1, 1].get_lines()[2].set_label('1-hour')
    axes[1, 1].legend()
    
    # Power Output
    axes[2, 0].hist(df_10min['power_w'].dropna(), bins=50, alpha=0.5, label='10-minute')
    axes[2, 0].hist(df_1h['power_w'].dropna(), bins=50, alpha=0.5, label='1-hour')
    axes[2, 0].set_title('Power Output Distribution')
    axes[2, 0].set_xlabel('Power [W]')
    axes[2, 0].set_ylabel('Frequency')
    axes[2, 0].legend()
    
    # Q-Q plot for Power
    axes[2, 1].set_title('Q-Q Plot: Power Output')
    stats.probplot(df_10min['power_w'].dropna(), dist="norm", plot=axes[2, 1])
    axes[2, 1].get_lines()[0].set_markerfacecolor('blue')
    axes[2, 1].get_lines()[0].set_label('10-minute')
    
    stats.probplot(df_1h['power_w'].dropna(), dist="norm", plot=axes[2, 1])
    axes[2, 1].get_lines()[2].set_markerfacecolor('red')
    axes[2, 1].get_lines()[2].set_label('1-hour')
    axes[2, 1].legend()
    
    plt.tight_layout()
    plt.savefig('visualizations/resolution_comparison/distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Autocorrelation Differences
    print("Analyzing autocorrelation differences...")
    
    # Calculate autocorrelation for both resolutions
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Global Radiation
    plot_acf(df_10min['GlobalRadiation [W m-2]'].dropna(), lags=50, ax=axes[0, 0])
    axes[0, 0].set_title('ACF: Global Radiation (10-minute)')
    
    plot_acf(df_1h['GlobalRadiation [W m-2]'].dropna(), lags=50, ax=axes[0, 1])
    axes[0, 1].set_title('ACF: Global Radiation (1-hour)')
    
    # Temperature
    plot_acf(df_10min['Temperature [degree_Celsius]'].dropna(), lags=50, ax=axes[1, 0])
    axes[1, 0].set_title('ACF: Temperature (10-minute)')
    
    plot_acf(df_1h['Temperature [degree_Celsius]'].dropna(), lags=50, ax=axes[1, 1])
    axes[1, 1].set_title('ACF: Temperature (1-hour)')
    
    # Power Output
    plot_acf(df_10min['power_w'].dropna(), lags=50, ax=axes[2, 0])
    axes[2, 0].set_title('ACF: Power Output (10-minute)')
    
    plot_acf(df_1h['power_w'].dropna(), lags=50, ax=axes[2, 1])
    axes[2, 1].set_title('ACF: Power Output (1-hour)')
    
    plt.tight_layout()
    plt.savefig('visualizations/resolution_comparison/autocorrelation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Spectral Analysis
    print("Performing spectral analysis...")
    
    # Calculate power spectral density for both resolutions
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Global Radiation
    f_10min, Pxx_10min = signal.welch(df_10min['GlobalRadiation [W m-2]'].dropna(), fs=6, nperseg=1024)
    axes[0, 0].semilogy(f_10min, Pxx_10min)
    axes[0, 0].set_title('PSD: Global Radiation (10-minute)')
    axes[0, 0].set_xlabel('Frequency [Hz]')
    axes[0, 0].set_ylabel('Power Spectral Density')
    
    f_1h, Pxx_1h = signal.welch(df_1h['GlobalRadiation [W m-2]'].dropna(), fs=1, nperseg=1024)
    axes[0, 1].semilogy(f_1h, Pxx_1h)
    axes[0, 1].set_title('PSD: Global Radiation (1-hour)')
    axes[0, 1].set_xlabel('Frequency [Hz]')
    axes[0, 1].set_ylabel('Power Spectral Density')
    
    # Temperature
    f_10min, Pxx_10min = signal.welch(df_10min['Temperature [degree_Celsius]'].dropna(), fs=6, nperseg=1024)
    axes[1, 0].semilogy(f_10min, Pxx_10min)
    axes[1, 0].set_title('PSD: Temperature (10-minute)')
    axes[1, 0].set_xlabel('Frequency [Hz]')
    axes[1, 0].set_ylabel('Power Spectral Density')
    
    f_1h, Pxx_1h = signal.welch(df_1h['Temperature [degree_Celsius]'].dropna(), fs=1, nperseg=1024)
    axes[1, 1].semilogy(f_1h, Pxx_1h)
    axes[1, 1].set_title('PSD: Temperature (1-hour)')
    axes[1, 1].set_xlabel('Frequency [Hz]')
    axes[1, 1].set_ylabel('Power Spectral Density')
    
    # Power Output
    f_10min, Pxx_10min = signal.welch(df_10min['power_w'].dropna(), fs=6, nperseg=1024)
    axes[2, 0].semilogy(f_10min, Pxx_10min)
    axes[2, 0].set_title('PSD: Power Output (10-minute)')
    axes[2, 0].set_xlabel('Frequency [Hz]')
    axes[2, 0].set_ylabel('Power Spectral Density')
    
    f_1h, Pxx_1h = signal.welch(df_1h['power_w'].dropna(), fs=1, nperseg=1024)
    axes[2, 1].semilogy(f_1h, Pxx_1h)
    axes[2, 1].set_title('PSD: Power Output (1-hour)')
    axes[2, 1].set_xlabel('Frequency [Hz]')
    axes[2, 1].set_ylabel('Power Spectral Density')
    
    plt.tight_layout()
    plt.savefig('visualizations/resolution_comparison/spectral_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Variability Analysis
    print("Analyzing variability differences...")
    
    # Calculate variability metrics
    # For 10-minute data, calculate the standard deviation within each hour
    hourly_std_10min = df_10min.groupby(pd.Grouper(freq='1H')).std()
    
    # For 1-hour data, calculate the standard deviation for each hour across days
    hourly_std_1h = df_1h.groupby(df_1h.index.hour).std()
    
    # Plot the distribution of hourly standard deviations
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Global Radiation
    axes[0].boxplot([hourly_std_10min['GlobalRadiation [W m-2]'].dropna(), 
                     hourly_std_1h['GlobalRadiation [W m-2]'].dropna()],
                    labels=['10-minute', '1-hour'])
    axes[0].set_title('Variability: Global Radiation')
    axes[0].set_ylabel('Standard Deviation [W/m²]')
    
    # Temperature
    axes[1].boxplot([hourly_std_10min['Temperature [degree_Celsius]'].dropna(), 
                     hourly_std_1h['Temperature [degree_Celsius]'].dropna()],
                    labels=['10-minute', '1-hour'])
    axes[1].set_title('Variability: Temperature')
    axes[1].set_ylabel('Standard Deviation [°C]')
    
    # Power Output
    axes[2].boxplot([hourly_std_10min['power_w'].dropna(), 
                     hourly_std_1h['power_w'].dropna()],
                    labels=['10-minute', '1-hour'])
    axes[2].set_title('Variability: Power Output')
    axes[2].set_ylabel('Standard Deviation [W]')
    
    plt.tight_layout()
    plt.savefig('visualizations/resolution_comparison/variability_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Implications for Deep Learning
    print("Creating visualizations for deep learning implications...")
    
    # Calculate the rate of change (first derivative) for both resolutions
    df_10min_diff = df_10min.diff()
    df_1h_diff = df_1h.diff()
    
    # Plot the distribution of rate of change
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Global Radiation
    axes[0, 0].hist(df_10min_diff['GlobalRadiation [W m-2]'].dropna(), bins=50, alpha=0.7)
    axes[0, 0].set_title('Rate of Change: Global Radiation (10-minute)')
    axes[0, 0].set_xlabel('Change in Global Radiation [W/m²]')
    axes[0, 0].set_ylabel('Frequency')
    
    axes[0, 1].hist(df_1h_diff['GlobalRadiation [W m-2]'].dropna(), bins=50, alpha=0.7)
    axes[0, 1].set_title('Rate of Change: Global Radiation (1-hour)')
    axes[0, 1].set_xlabel('Change in Global Radiation [W/m²]')
    axes[0, 1].set_ylabel('Frequency')
    
    # Temperature
    axes[1, 0].hist(df_10min_diff['Temperature [degree_Celsius]'].dropna(), bins=50, alpha=0.7)
    axes[1, 0].set_title('Rate of Change: Temperature (10-minute)')
    axes[1, 0].set_xlabel('Change in Temperature [°C]')
    axes[1, 0].set_ylabel('Frequency')
    
    axes[1, 1].hist(df_1h_diff['Temperature [degree_Celsius]'].dropna(), bins=50, alpha=0.7)
    axes[1, 1].set_title('Rate of Change: Temperature (1-hour)')
    axes[1, 1].set_xlabel('Change in Temperature [°C]')
    axes[1, 1].set_ylabel('Frequency')
    
    # Power Output
    axes[2, 0].hist(df_10min_diff['power_w'].dropna(), bins=50, alpha=0.7)
    axes[2, 0].set_title('Rate of Change: Power Output (10-minute)')
    axes[2, 0].set_xlabel('Change in Power [W]')
    axes[2, 0].set_ylabel('Frequency')
    
    axes[2, 1].hist(df_1h_diff['power_w'].dropna(), bins=50, alpha=0.7)
    axes[2, 1].set_title('Rate of Change: Power Output (1-hour)')
    axes[2, 1].set_xlabel('Change in Power [W]')
    axes[2, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('visualizations/resolution_comparison/rate_of_change.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print('All temporal resolution comparison visualizations have been created and saved to the visualizations directory.')

if __name__ == "__main__":
    create_resolution_comparison_visualizations()