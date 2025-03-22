#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Station Data Analysis and Visualization Script

This script analyzes the station data parquet files and creates professional
visualizations suitable for scientific papers. It focuses on:
1. Time series visualization of key variables
2. Statistical distributions
3. Correlation analysis
4. Daily/seasonal patterns
5. Feature importance for deep learning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from pandas.plotting import autocorrelation_plot, lag_plot

def create_visualizations():
    """Create professional visualizations from station data."""
    # Create visualizations directory if it doesn't exist
    os.makedirs('visualizations/time_series', exist_ok=True)
    os.makedirs('visualizations/distributions', exist_ok=True)
    os.makedirs('visualizations/correlations', exist_ok=True)
    os.makedirs('visualizations/patterns', exist_ok=True)

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

    # 1. Time Series Visualization - Key Variables
    print("Creating time series visualizations...")
    # Sample one week of data for detailed visualization
    start_date = '2022-08-01'
    end_date = '2022-08-07'
    df_week_10min = df_10min.loc[start_date:end_date]
    df_week_1h = df_1h.loc[start_date:end_date]

    # Plot 1: Solar radiation and power output (1-hour data)
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

    plt.title('Solar Radiation and PV Power Output (Hourly Data)')
    plt.tight_layout()
    plt.savefig('visualizations/time_series/radiation_power_hourly.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Clear sky vs actual radiation (10-min data)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_week_10min.index, df_week_10min['GlobalRadiation [W m-2]'], 'b-', alpha=0.7, label='Measured Radiation')
    ax.plot(df_week_10min.index, df_week_10min['ClearSkyGHI'], 'r--', alpha=0.7, label='Clear Sky GHI')
    ax.set_xlabel('Date')
    ax.set_ylabel('Radiation [W/m²]')
    ax.set_title('Measured vs Clear Sky Radiation (10-minute Data)')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    plt.tight_layout()
    plt.savefig('visualizations/time_series/measured_vs_clearsky.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Statistical Distributions
    print("Creating statistical distribution plots...")
    # Plot 3: Distribution of key variables
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Global Radiation
    axes[0, 0].hist(df_1h['GlobalRadiation [W m-2]'].dropna(), bins=50, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Distribution of Global Radiation')
    axes[0, 0].set_xlabel('Global Radiation [W/m²]')
    axes[0, 0].set_ylabel('Frequency')

    # Temperature
    axes[0, 1].hist(df_1h['Temperature [degree_Celsius]'].dropna(), bins=50, color='salmon', alpha=0.7)
    axes[0, 1].set_title('Distribution of Temperature')
    axes[0, 1].set_xlabel('Temperature [°C]')
    axes[0, 1].set_ylabel('Frequency')

    # Wind Speed
    axes[1, 0].hist(df_1h['WindSpeed [m s-1]'].dropna(), bins=50, color='lightgreen', alpha=0.7)
    axes[1, 0].set_title('Distribution of Wind Speed')
    axes[1, 0].set_xlabel('Wind Speed [m/s]')
    axes[1, 0].set_ylabel('Frequency')

    # Power Output
    axes[1, 1].hist(df_1h['power_w'].dropna(), bins=50, color='purple', alpha=0.7)
    axes[1, 1].set_title('Distribution of Power Output')
    axes[1, 1].set_xlabel('Power [W]')
    axes[1, 1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('visualizations/distributions/key_variables_dist.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Correlation Analysis
    print("Creating correlation analysis plots...")
    # Plot 4: Correlation matrix of key variables
    key_vars = ['GlobalRadiation [W m-2]', 'Temperature [degree_Celsius]', 'WindSpeed [m s-1]', 
                'ClearSkyGHI', 'ClearSkyIndex', 'power_w']
    corr = df_1h[key_vars].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)

    # Add correlation values
    for i in range(len(corr)):
        for j in range(len(corr)):
            text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                        ha='center', va='center', color='black')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr.columns)

    plt.title('Correlation Matrix of Key Variables')
    plt.tight_layout()
    plt.savefig('visualizations/correlations/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 5: Scatter plot of radiation vs power
    plt.figure(figsize=(10, 8))
    plt.scatter(df_1h['GlobalRadiation [W m-2]'], df_1h['power_w'], alpha=0.3, color='blue')

    # Add regression line - ensure same indices by using dropna on the DataFrame
    df_regression = df_1h[['GlobalRadiation [W m-2]', 'power_w']].dropna()
    if len(df_regression) > 0:
        z = np.polyfit(df_regression['GlobalRadiation [W m-2]'], df_regression['power_w'], 1)
        p = np.poly1d(z)
        plt.plot(np.sort(df_regression['GlobalRadiation [W m-2]']),
                p(np.sort(df_regression['GlobalRadiation [W m-2]'])),
                'r--', linewidth=2)

    plt.title('Relationship Between Solar Radiation and Power Output')
    plt.xlabel('Global Radiation [W/m²]')
    plt.ylabel('Power Output [W]')
    plt.tight_layout()
    plt.savefig('visualizations/correlations/radiation_power_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Daily and Seasonal Patterns
    print("Creating daily and seasonal pattern plots...")
    # Plot 6: Daily patterns of radiation and power
    df_1h['hour'] = df_1h.index.hour
    hourly_avg = df_1h.groupby('hour').agg({
        'GlobalRadiation [W m-2]': 'mean',
        'power_w': 'mean',
        'Temperature [degree_Celsius]': 'mean'
    }).reset_index()

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Global Radiation [W/m²]', color=color)
    ax1.plot(hourly_avg['hour'], hourly_avg['GlobalRadiation [W m-2]'], color=color, marker='o', label='Global Radiation')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Power [W]', color=color)
    ax2.plot(hourly_avg['hour'], hourly_avg['power_w'], color=color, marker='s', label='Power Output')
    ax2.tick_params(axis='y', labelcolor=color)

    # Add combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title('Average Daily Pattern of Solar Radiation and Power Output')
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig('visualizations/patterns/daily_pattern.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 7: Seasonal patterns - Monthly averages
    df_1h['month'] = df_1h.index.month
    monthly_avg = df_1h.groupby('month').agg({
        'GlobalRadiation [W m-2]': 'mean',
        'power_w': 'mean',
        'Temperature [degree_Celsius]': 'mean'
    }).reset_index()

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Global Radiation [W/m²]', color=color)
    ax1.plot(monthly_avg['month'], monthly_avg['GlobalRadiation [W m-2]'], color=color, marker='o', label='Global Radiation')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Power [W]', color=color)
    ax2.plot(monthly_avg['month'], monthly_avg['power_w'], color=color, marker='s', label='Power Output')
    ax2.tick_params(axis='y', labelcolor=color)

    # Add combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title('Monthly Average of Solar Radiation and Power Output')
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.tight_layout()
    plt.savefig('visualizations/patterns/monthly_pattern.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Feature Importance Analysis for Deep Learning
    print("Creating feature importance analysis plots...")
    # Plot 8: Feature correlation with target variable (power_w)
    target_corr = df_1h.corr()['power_w'].sort_values(ascending=False).drop('power_w')
    plt.figure(figsize=(12, 8))
    plt.bar(target_corr.index, target_corr.values, color='skyblue')
    plt.title('Feature Correlation with Power Output')
    plt.xlabel('Features')
    plt.ylabel('Correlation Coefficient')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('visualizations/correlations/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 9: Clear Sky Index vs Power Output with Temperature coloring
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(df_1h['ClearSkyIndex'], df_1h['power_w'], 
                    c=df_1h['Temperature [degree_Celsius]'], 
                    alpha=0.3, cmap='viridis')
    plt.colorbar(sc, label='Temperature [°C]')
    plt.title('Clear Sky Index vs Power Output (Colored by Temperature)')
    plt.xlabel('Clear Sky Index')
    plt.ylabel('Power Output [W]')
    plt.tight_layout()
    plt.savefig('visualizations/correlations/clearsky_index_power.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 10: Time series decomposition - Daily average power output over time
    print("Creating time series analysis plots...")
    df_daily = df_1h.resample('D').mean()

    plt.figure(figsize=(14, 6))
    plt.plot(df_daily.index, df_daily['power_w'], 'b-')
    plt.title('Daily Average Power Output Over Time')
    plt.xlabel('Date')
    plt.ylabel('Power Output [W]')
    plt.tight_layout()
    plt.savefig('visualizations/time_series/power_time_series.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Additional plot: Autocorrelation analysis for time series
    plt.figure(figsize=(12, 6))
    autocorrelation_plot(df_daily['power_w'].dropna())
    plt.title('Autocorrelation of Daily Power Output')
    plt.tight_layout()
    plt.savefig('visualizations/time_series/power_autocorrelation.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Additional plot: Lag plot for time series analysis
    plt.figure(figsize=(10, 10))
    lag_plot(df_daily['power_w'].dropna(), lag=1)
    plt.title('Lag Plot of Daily Power Output (Lag=1)')
    plt.tight_layout()
    plt.savefig('visualizations/time_series/power_lag_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Additional plot: Temporal resolution comparison (10min vs 1h)
    # Select a single day for comparison
    sample_day = '2022-08-01'
    df_day_10min = df_10min.loc[sample_day]
    df_day_1h = df_1h.loc[sample_day]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_day_10min.index, df_day_10min['GlobalRadiation [W m-2]'], 'b-', alpha=0.5, label='10-minute Resolution')
    ax.plot(df_day_1h.index, df_day_1h['GlobalRadiation [W m-2]'], 'r-o', alpha=0.8, label='1-hour Resolution')
    ax.set_xlabel('Time')
    ax.set_ylabel('Global Radiation [W/m²]')
    ax.set_title('Comparison of Temporal Resolutions (10min vs 1h) - August 1, 2022')
    ax.legend()
    plt.tight_layout()
    plt.savefig('visualizations/time_series/resolution_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Additional plot: Heatmap of hourly power output by month
    print("Creating advanced visualization plots...")
    # Create pivot table for month vs hour heatmap
    df_1h['month'] = df_1h.index.month
    df_1h['hour'] = df_1h.index.hour
    pivot = df_1h.pivot_table(values='power_w', index='month', columns='hour', aggfunc='mean')
    
    plt.figure(figsize=(14, 8))
    plt.imshow(pivot, cmap='viridis', aspect='auto', interpolation='nearest')
    plt.colorbar(label='Average Power Output [W]')
    plt.title('Heatmap of Average Power Output by Month and Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Month')
    plt.xticks(range(0, 24))
    plt.yticks(range(0, 12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.tight_layout()
    plt.savefig('visualizations/patterns/power_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Additional plot: Boxplot of power output by hour
    plt.figure(figsize=(14, 8))
    boxplot_data = [df_1h[df_1h['hour'] == h]['power_w'].dropna() for h in range(24)]
    plt.boxplot(boxplot_data, patch_artist=True, 
                boxprops=dict(facecolor='lightblue', color='blue'),
                whiskerprops=dict(color='blue'),
                medianprops=dict(color='red'))
    plt.title('Distribution of Power Output by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Power Output [W]')
    plt.xticks(range(1, 25), range(0, 24))
    plt.tight_layout()
    plt.savefig('visualizations/distributions/power_boxplot_hourly.png', dpi=300, bbox_inches='tight')
    plt.close()

    print('All visualizations have been created and saved to the visualizations directory.')

if __name__ == "__main__":
    create_visualizations()