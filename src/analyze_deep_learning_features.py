#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Deep Learning Feature Analysis for Time Series Data

This script analyzes the station data from a deep learning perspective,
focusing on features and characteristics relevant for time series models
like LSTM. It creates visualizations suitable for scientific papers.

Key analyses:
1. Time series decomposition
2. Feature importance for LSTM models
3. Sequence visualization
4. Autocorrelation and stationarity
5. Temporal patterns relevant for deep learning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_regression

def create_deep_learning_visualizations():
    """Create visualizations focused on deep learning for time series data."""
    # Create visualizations directory if it doesn't exist
    os.makedirs('visualizations/deep_learning', exist_ok=True)
    os.makedirs('visualizations/time_series_analysis', exist_ok=True)
    os.makedirs('visualizations/feature_importance', exist_ok=True)
    os.makedirs('visualizations/sequence_analysis', exist_ok=True)

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

    # 1. Time Series Decomposition
    print("Performing time series decomposition...")
    # Resample to daily for better visualization of seasonal patterns
    df_daily = df_1h.resample('D').mean()
    
    # Fill any missing values for decomposition
    power_series = df_daily['power_w'].fillna(method='ffill')
    
    # Perform seasonal decomposition
    try:
        # Try with period=365 for annual seasonality
        decomposition = seasonal_decompose(power_series, model='additive', period=365)
        
        # Plot decomposition
        fig, axes = plt.subplots(4, 1, figsize=(12, 16))
        
        # Original
        axes[0].plot(decomposition.observed)
        axes[0].set_title('Original Time Series')
        axes[0].set_ylabel('Power [W]')
        
        # Trend
        axes[1].plot(decomposition.trend)
        axes[1].set_title('Trend Component')
        axes[1].set_ylabel('Power [W]')
        
        # Seasonal
        axes[2].plot(decomposition.seasonal)
        axes[2].set_title('Seasonal Component')
        axes[2].set_ylabel('Power [W]')
        
        # Residual
        axes[3].plot(decomposition.resid)
        axes[3].set_title('Residual Component')
        axes[3].set_ylabel('Power [W]')
        
        plt.tight_layout()
        plt.savefig('visualizations/time_series_analysis/seasonal_decomposition_annual.png', dpi=300, bbox_inches='tight')
        plt.close()
    except:
        print("Annual decomposition failed, trying with weekly seasonality...")
        
    # Try with period=7 for weekly seasonality
    try:
        decomposition = seasonal_decompose(power_series, model='additive', period=7)
        
        # Plot decomposition
        fig, axes = plt.subplots(4, 1, figsize=(12, 16))
        
        # Original
        axes[0].plot(decomposition.observed)
        axes[0].set_title('Original Time Series')
        axes[0].set_ylabel('Power [W]')
        
        # Trend
        axes[1].plot(decomposition.trend)
        axes[1].set_title('Trend Component')
        axes[1].set_ylabel('Power [W]')
        
        # Seasonal
        axes[2].plot(decomposition.seasonal)
        axes[2].set_title('Seasonal Component')
        axes[2].set_ylabel('Power [W]')
        
        # Residual
        axes[3].plot(decomposition.resid)
        axes[3].set_title('Residual Component')
        axes[3].set_ylabel('Power [W]')
        
        plt.tight_layout()
        plt.savefig('visualizations/time_series_analysis/seasonal_decomposition_weekly.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Weekly decomposition failed: {e}")

    # 2. Feature Importance for LSTM Models
    print("Analyzing feature importance for deep learning...")
    
    # Calculate mutual information between features and target
    # This is a non-linear measure of dependency, suitable for deep learning
    X = df_1h.drop(['power_w', 'energy_wh', 'energy_interval'], axis=1).select_dtypes(include=[np.number])
    y = df_1h['power_w']
    
    # Handle any missing values
    X = X.fillna(0)
    y = y.fillna(0)
    
    # Calculate mutual information
    mi_scores = mutual_info_regression(X, y)
    mi_scores = pd.Series(mi_scores, index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    
    # Plot mutual information scores
    plt.figure(figsize=(12, 8))
    mi_scores.plot(kind='bar', color='skyblue')
    plt.title('Feature Importance (Mutual Information) for Power Output Prediction')
    plt.xlabel('Features')
    plt.ylabel('Mutual Information Score')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance/mutual_information.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Sequence Visualization (important for LSTM)
    print("Creating sequence visualizations for LSTM...")
    
    # Create a sample sequence (24 hours) for visualization
    sample_date = '2022-08-01'
    sequence_length = 24
    
    # Get a sequence of data
    sequence_data = df_1h.loc[sample_date:].iloc[:sequence_length]
    
    # Plot key features in the sequence
    fig, axes = plt.subplots(4, 1, figsize=(12, 16))
    
    # Global Radiation
    axes[0].plot(sequence_data.index, sequence_data['GlobalRadiation [W m-2]'], 'b-')
    axes[0].set_title('Global Radiation Sequence')
    axes[0].set_ylabel('Global Radiation [W/m²]')
    
    # Temperature
    axes[1].plot(sequence_data.index, sequence_data['Temperature [degree_Celsius]'], 'r-')
    axes[1].set_title('Temperature Sequence')
    axes[1].set_ylabel('Temperature [°C]')
    
    # Clear Sky Index
    axes[2].plot(sequence_data.index, sequence_data['ClearSkyIndex'], 'g-')
    axes[2].set_title('Clear Sky Index Sequence')
    axes[2].set_ylabel('Clear Sky Index')
    
    # Power Output
    axes[3].plot(sequence_data.index, sequence_data['power_w'], 'purple')
    axes[3].set_title('Power Output Sequence (Target Variable)')
    axes[3].set_ylabel('Power [W]')
    
    plt.tight_layout()
    plt.savefig('visualizations/sequence_analysis/sequence_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Autocorrelation and Partial Autocorrelation Analysis
    print("Performing autocorrelation analysis...")
    
    # Calculate ACF and PACF for power output
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # ACF
    plot_acf(df_daily['power_w'].dropna(), lags=50, ax=axes[0])
    axes[0].set_title('Autocorrelation Function (ACF) for Power Output')
    
    # PACF
    plot_pacf(df_daily['power_w'].dropna(), lags=50, ax=axes[1])
    axes[1].set_title('Partial Autocorrelation Function (PACF) for Power Output')
    
    plt.tight_layout()
    plt.savefig('visualizations/time_series_analysis/acf_pacf.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Stationarity Test (important for time series modeling)
    print("Testing stationarity of time series...")
    
    # Perform Augmented Dickey-Fuller test
    result = adfuller(df_daily['power_w'].dropna())
    
    # Create a table with the results
    adf_output = pd.Series({
        'Test Statistic': result[0],
        'p-value': result[1],
        'Lags Used': result[2],
        'Number of Observations': result[3],
        'Critical Value (1%)': result[4]['1%'],
        'Critical Value (5%)': result[4]['5%'],
        'Critical Value (10%)': result[4]['10%']
    })
    
    # Plot the results as a table
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=[[v] for v in adf_output.values],
             rowLabels=adf_output.index,
             colLabels=["Value"],
             cellLoc='center',
             loc='center')
    
    plt.title('Augmented Dickey-Fuller Test Results for Power Output')
    plt.tight_layout()
    plt.savefig('visualizations/time_series_analysis/stationarity_test.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Rolling Statistics (mean and std) - important for detecting non-stationarity
    print("Calculating rolling statistics...")
    
    # Calculate rolling statistics
    rolling_mean = df_daily['power_w'].rolling(window=30).mean()
    rolling_std = df_daily['power_w'].rolling(window=30).std()
    
    plt.figure(figsize=(12, 6))
    plt.plot(df_daily.index, df_daily['power_w'], label='Original')
    plt.plot(rolling_mean.index, rolling_mean, label='Rolling Mean (30 days)')
    plt.plot(rolling_std.index, rolling_std, label='Rolling Std (30 days)')
    plt.title('Rolling Statistics for Power Output')
    plt.xlabel('Date')
    plt.ylabel('Power [W]')
    plt.legend()
    plt.tight_layout()
    plt.savefig('visualizations/time_series_analysis/rolling_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Visualize normalized sequences (as would be fed to LSTM)
    print("Creating normalized sequence visualizations...")
    
    # Normalize the sequence data
    scaler = MinMaxScaler()
    sequence_scaled = scaler.fit_transform(sequence_data[['GlobalRadiation [W m-2]', 'Temperature [degree_Celsius]', 
                                                         'ClearSkyIndex', 'power_w']])
    
    # Convert back to DataFrame for plotting
    sequence_scaled_df = pd.DataFrame(sequence_scaled, 
                                     index=sequence_data.index,
                                     columns=['Global Radiation', 'Temperature', 'Clear Sky Index', 'Power Output'])
    
    # Plot normalized sequences
    plt.figure(figsize=(12, 6))
    for column in sequence_scaled_df.columns:
        plt.plot(sequence_scaled_df.index, sequence_scaled_df[column], label=column)
    
    plt.title('Normalized Input Sequence for LSTM (Min-Max Scaling)')
    plt.xlabel('Time')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig('visualizations/deep_learning/normalized_sequence.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. Visualize the circular time features (sin/cos encoding)
    print("Visualizing circular time features...")
    
    # Plot hour sin/cos encoding
    plt.figure(figsize=(12, 6))
    plt.scatter(df_1h['hour_sin'], df_1h['hour_cos'], c=df_1h.index.hour, cmap='hsv', alpha=0.5)
    plt.colorbar(label='Hour of Day')
    plt.title('Circular Encoding of Hour (sin/cos)')
    plt.xlabel('sin(hour)')
    plt.ylabel('cos(hour)')
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('visualizations/deep_learning/hour_circular_encoding.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot day sin/cos encoding
    plt.figure(figsize=(12, 6))
    plt.scatter(df_1h['day_sin'], df_1h['day_cos'], c=df_1h.index.dayofyear, cmap='hsv', alpha=0.5)
    plt.colorbar(label='Day of Year')
    plt.title('Circular Encoding of Day of Year (sin/cos)')
    plt.xlabel('sin(day)')
    plt.ylabel('cos(day)')
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('visualizations/deep_learning/day_circular_encoding.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 9. Visualize the relationship between time features and power output
    print("Analyzing time feature relationships...")
    
    # Hour of day vs power output
    hourly_stats = df_1h.groupby(df_1h.index.hour)['power_w'].agg(['mean', 'std']).reset_index()
    hourly_stats.columns = ['Hour', 'Mean Power', 'Std Power']
    
    plt.figure(figsize=(12, 6))
    plt.errorbar(hourly_stats['Hour'], hourly_stats['Mean Power'], 
                yerr=hourly_stats['Std Power'], fmt='o-', capsize=5)
    plt.title('Power Output by Hour of Day (with Standard Deviation)')
    plt.xlabel('Hour of Day')
    plt.ylabel('Power Output [W]')
    plt.xticks(range(0, 24))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('visualizations/deep_learning/hour_power_relationship.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Month vs power output
    monthly_stats = df_1h.groupby(df_1h.index.month)['power_w'].agg(['mean', 'std']).reset_index()
    monthly_stats.columns = ['Month', 'Mean Power', 'Std Power']
    
    plt.figure(figsize=(12, 6))
    plt.errorbar(monthly_stats['Month'], monthly_stats['Mean Power'], 
                yerr=monthly_stats['Std Power'], fmt='o-', capsize=5)
    plt.title('Power Output by Month (with Standard Deviation)')
    plt.xlabel('Month')
    plt.ylabel('Power Output [W]')
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('visualizations/deep_learning/month_power_relationship.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print('All deep learning visualizations have been created and saved to the visualizations directory.')

if __name__ == "__main__":
    create_deep_learning_visualizations()