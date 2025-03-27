import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler
import os

# Create results directory if it doesn't exist
if not os.path.exists('results/feature_analysis'):
    os.makedirs('results/feature_analysis')

# Load the data
df = pd.read_parquet('data/station_data_1h.parquet')

# Basic info about the dataset
print("Dataset shape:", df.shape)
print("\nMissing values per column:")
print(df.isna().sum())

# Check for correlation with power_w
correlation = df.corr()['power_w'].sort_values(ascending=False)
print("\nCorrelation with power_w:")
print(correlation)

# Save correlation to CSV
correlation.to_csv('results/feature_analysis/correlation_with_power_w.csv')

# Create correlation heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('results/feature_analysis/correlation_heatmap.png')

# Calculate mutual information (for non-linear relationships)
def calculate_mi(X, y):
    mi = mutual_info_regression(X, y)
    mi_series = pd.Series(mi, index=X.columns)
    return mi_series.sort_values(ascending=False)

# Fill NAs before calculating MI
df_for_mi = df.fillna(method='ffill').fillna(method='bfill')
X = df_for_mi.drop(columns=['power_w', 'energy_wh', 'energy_interval'])
y = df_for_mi['power_w']

mi_scores = calculate_mi(X, y)
print("\nMutual Information Scores:")
print(mi_scores)

# Save MI scores to CSV
mi_scores.to_csv('results/feature_analysis/mutual_info_scores.csv')

# Plot MI scores
plt.figure(figsize=(12, 8))
mi_scores.sort_values().plot(kind='barh')
plt.title('Mutual Information Scores for power_w Prediction')
plt.tight_layout()
plt.savefig('results/feature_analysis/mutual_info_scores.png')

# Plot time series distribution of key features
plt.figure(figsize=(15, 10))
plt.subplot(4, 1, 1)
plt.plot(df.index, df['power_w'])
plt.title('Power Output (W)')
plt.subplot(4, 1, 2)
plt.plot(df.index, df['GlobalRadiation [W m-2]'])
plt.title('Global Radiation (W/m²)')
plt.subplot(4, 1, 3)
plt.plot(df.index, df['Temperature [degree_Celsius]'])
plt.title('Temperature (°C)')
plt.subplot(4, 1, 4)
plt.plot(df.index, df['ClearSkyIndex'])
plt.title('Clear Sky Index')
plt.tight_layout()
plt.savefig('results/feature_analysis/time_series_plots.png')

# Check for multi-collinearity and remove redundant features
print("\nFeatures with high correlation (>0.9):")
corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
redundant = [column for column in upper.columns if any(upper[column] > 0.9)]
print(redundant)

# Check for daily and seasonal patterns in power output
df['hour'] = df.index.hour
df['month'] = df.index.month

plt.figure(figsize=(12, 6))
sns.boxplot(x='hour', y='power_w', data=df)
plt.title('Power Output Distribution by Hour')
plt.savefig('results/feature_analysis/power_by_hour.png')

plt.figure(figsize=(12, 6))
sns.boxplot(x='month', y='power_w', data=df)
plt.title('Power Output Distribution by Month')
plt.savefig('results/feature_analysis/power_by_month.png')

# Create and save the dataset with selected features
def create_feature_dataset(feature_list):
    selected_df = df[feature_list + ['power_w']]
    output_path = 'data/lstm_features.parquet'
    selected_df.to_parquet(output_path)
    print(f"Dataset with selected features saved to {output_path}")
    print(f"Features included: {feature_list}")
    return selected_df

# Create a version with key weather and time features
# We'll define this after analyzing results

print("Analysis complete. Check the results/feature_analysis directory for output files.")