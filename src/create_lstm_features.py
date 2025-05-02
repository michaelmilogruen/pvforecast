import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression
import os

# Load the original dataset
print("Loading station data...")
df = pd.read_parquet('data/station_data_1h.parquet')

# Print basic info
print(f"Original dataset shape: {df.shape}")
print(f"Missing values: {df.isna().sum().sum()}")

# Based on correlation analysis and mutual information scores:
# 1. GlobalRadiation has the highest correlation with power_w
# 2. Several ClearSky features are highly correlated with each other
# 3. Temporal features (hour_sin, hour_cos, day_sin, day_cos) capture cyclical patterns
# 4. Temperature and WindSpeed have moderate correlation

# Primary features with high importance
key_features = [
    'GlobalRadiation [W m-2]',  # Highest correlation and MI score
    'ClearSkyGHI',              # Strong indicator, less redundant than DNI/DHI
    'ClearSkyIndex',            # Provides ratio information
    'isNight',                  # Important binary feature
    'Temperature [degree_Celsius]',  # Moderate correlation, affects panel efficiency
    'hour_sin', 'hour_cos',     # Diurnal patterns
    'day_sin', 'day_cos'        # Seasonal patterns
]

# Additional weather features that may be useful
secondary_features = [
    'WindSpeed [m s-1]',        # Moderate correlation, affects panel cooling
]

# Combine all selected features
selected_features = key_features + secondary_features

# Add target variable
selected_columns = selected_features + ['power_w']

# Create a new dataframe with selected features
print(f"Creating new dataset with {len(selected_features)} features...")
selected_df = df[selected_columns].copy()

# Handle missing values (usually simple forward-fill is good for time series)
selected_df = selected_df.ffill().bfill()

# Verify no missing values remain
print(f"Missing values after filling: {selected_df.isna().sum().sum()}")

# Save to parquet file
output_path = 'data/lstm_features.parquet'
selected_df.to_parquet(output_path)
print(f"Dataset saved to {output_path}")

# Print feature list for reference
print("Selected features:")
for feature in selected_features:
    print(f"- {feature}")

# Optional: Create a visualization of the selected features
plt.figure(figsize=(12, 8))
correlation = selected_df.corr()['power_w'].sort_values(ascending=False)
correlation.drop('power_w').plot(kind='barh')
plt.title('Correlation of Selected Features with power_w')
plt.tight_layout()

if not os.path.exists('results/feature_analysis'):
    os.makedirs('results/feature_analysis')
plt.savefig('results/feature_analysis/selected_features_correlation.png')
print("Feature selection visualization saved")

print("Feature selection complete")
