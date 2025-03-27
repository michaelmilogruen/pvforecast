"""
Create LSTM feature set with statistical aggregations from 10-minute data
This script:
1. Loads 10-minute resolution data
2. Computes statistical aggregations (mean, max, min, std) for each hour
3. Merges these aggregated features with the previously selected 1-hour features
4. Saves the result as a new parquet file with hourly timestamps
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Creating enhanced LSTM feature set with statistical aggregations...")

# Create directories for results
if not os.path.exists('results/feature_analysis'):
    os.makedirs('results/feature_analysis')

# Load datasets
print("Loading datasets...")
df_10min = pd.read_parquet('data/station_data_10min.parquet')
df_1h = pd.read_parquet('data/lstm_features.parquet')

print(f"10-minute data: {df_10min.shape} rows")
print(f"1-hour data: {df_1h.shape} rows")

# Define features to aggregate
# These are key weather and irradiance features that can benefit from 
# capturing sub-hourly variations
features_to_aggregate = [
    'GlobalRadiation [W m-2]',
    'Temperature [degree_Celsius]',
    'WindSpeed [m s-1]',
    'ClearSkyIndex',
    'Precipitation [mm]'
]

# Define aggregation functions for each feature
aggregation_dict = {
    'GlobalRadiation [W m-2]': ['mean', 'max', 'min', 'std'],
    'Temperature [degree_Celsius]': ['mean', 'max', 'min', 'std'],
    'WindSpeed [m s-1]': ['mean', 'max', 'std'],
    'ClearSkyIndex': ['mean', 'min', 'std'],
    'Precipitation [mm]': ['sum', 'max']
}

print(f"Computing statistical aggregations for {len(features_to_aggregate)} features...")

# Resample to hourly frequency with the specified aggregations
# First ensure the index is DatetimeIndex for proper resampling
df_10min = df_10min.sort_index()

# Create a new dataframe to store the aggregated features
agg_dfs = []

# For each feature, compute the specified aggregations
for feature, aggs in aggregation_dict.items():
    # Select just this feature to reduce memory usage during aggregation
    feature_df = df_10min[[feature]]
    
    # Create a dictionary mapping the feature to its aggregation functions
    agg_dict = {feature: aggs}
    
    # Resample to hourly frequency with the specified aggregations
    hourly_agg = feature_df.resample('1H').agg(agg_dict)
    
    # Flatten the column names
    hourly_agg.columns = [f"{feature}_{agg}" for agg in aggs]
    
    # Add to the list of aggregated dataframes
    agg_dfs.append(hourly_agg)

# Combine all aggregated features into a single dataframe
df_agg = pd.concat(agg_dfs, axis=1)

# Check for any missing values and handle them
if df_agg.isna().any().any():
    print(f"Handling {df_agg.isna().sum().sum()} missing values in aggregated data...")
    df_agg = df_agg.fillna(method='ffill').fillna(method='bfill')

print(f"Created {df_agg.shape[1]} aggregated features")

# Merge the aggregated features with the existing 1-hour features
# Ensure the indexes (timestamps) match by using the index from 1h data
df_agg = df_agg.reindex(df_1h.index)

# Check for any missing values after reindexing
if df_agg.isna().any().any():
    print(f"Handling {df_agg.isna().sum().sum()} missing values after reindexing...")
    df_agg = df_agg.fillna(method='ffill').fillna(method='bfill')

# Combine with the existing 1-hour data
df_combined = pd.concat([df_1h, df_agg], axis=1)

print(f"Final dataset contains {df_combined.shape[1]} features:")

# List all features for reference
for column in df_combined.columns:
    print(f"- {column}")

# Create visualization of feature correlations with power_w
plt.figure(figsize=(14, 10))
correlation = df_combined.corr()['power_w'].sort_values(ascending=False)
correlation = correlation.drop('power_w')

# Plot top 20 correlations
top_n = 20
correlation_top = correlation.head(top_n)
correlation_top.plot(kind='barh', figsize=(10, 8))
plt.title(f'Top {top_n} Features by Correlation with power_w')
plt.tight_layout()
plt.savefig('results/feature_analysis/top_correlations_with_aggregations.png')

# Save the final dataset
output_path = 'data/lstm_features_with_aggregations.parquet'
df_combined.to_parquet(output_path)
print(f"\nDataset with hourly features and sub-hourly statistical aggregations saved to {output_path}")

# Summary statistics
print("\nSummary statistics of the final dataset:")
print(df_combined.describe().T[['mean', 'std', 'min', 'max']])

print("\nFeature aggregation complete!")