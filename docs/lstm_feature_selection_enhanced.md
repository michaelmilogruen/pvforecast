# Enhanced LSTM Feature Selection for PV Power Prediction

## Overview

This document explains the enhanced feature selection process for the LSTM model that predicts `power_w` (PV power output). The enhanced features include both hourly data and statistical aggregations from 10-minute resolution data, saved in `data/lstm_features_with_aggregations.parquet`.

## Multi-Resolution Approach

The enhanced feature set combines:

1. **Hourly features**: Base predictors selected from 1-hour resolution data
2. **Statistical aggregations**: Mean, max, min, standard deviation, and other statistical measures derived from 10-minute resolution data

This multi-resolution approach captures important sub-hourly variations that a simple hourly average would miss, such as:

- **Radiation fluctuations**: Short-term cloud movements affecting solar panels
- **Temperature variability**: Thermal dynamics affecting panel efficiency
- **Wind patterns**: Brief gusts that might affect cooling or dust removal
- **Precipitation events**: Short rain events that might not be evident in hourly data

## Feature Set Description

### Base Hourly Features

| Feature | Selection Rationale | Correlation with power_w |
|---------|---------------------|--------------------------|
| GlobalRadiation [W m-2] | Primary driver of PV generation | 0.893 |
| ClearSkyGHI | Theoretical maximum irradiance | 0.827 |
| ClearSkyIndex | Ratio capturing cloud effects | 0.651 |
| isNight | Binary feature for daylight availability | -0.633 |
| Temperature [degree_Celsius] | Affects panel efficiency | 0.509 |
| WindSpeed [m s-1] | Affects panel cooling | 0.377 |
| hour_sin, hour_cos | Captures diurnal patterns cyclically | 0.124, -0.634 |
| day_sin, day_cos | Captures seasonal patterns cyclically | 0.020, -0.287 |

### Statistical Aggregations from 10-minute Data

| Feature | Aggregations | Rationale |
|---------|-------------|-----------|
| GlobalRadiation [W m-2] | mean, max, min, std | Captures radiation peaks and variability within each hour |
| Temperature [degree_Celsius] | mean, max, min, std | Captures temperature fluctuations affecting panel efficiency |
| WindSpeed [m s-1] | mean, max, std | Captures wind gusts and variability |
| ClearSkyIndex | mean, min, std | Captures cloud coverage patterns within each hour |
| Precipitation [mm] | sum, max | Captures brief rain events that affect panel performance |

## Benefits of Multi-Resolution Features

1. **Captures transient events**: Brief cloud cover, rain showers, or temperature fluctuations
2. **Provides variability metrics**: Standard deviation features indicate stability/instability
3. **Preserves extreme values**: Maximum and minimum values that would be smoothed out in hourly averages
4. **Improves model robustness**: Better handling of edge cases and unusual weather patterns

## Feature Importance

Initial analysis shows that several sub-hourly statistical features have high correlation with power output:

- `GlobalRadiation [W m-2]_max`: Highest radiation value within the hour
- `GlobalRadiation [W m-2]_std`: Variability of radiation within the hour
- `Temperature [degree_Celsius]_max`: Peak temperature affecting panel efficiency
- `ClearSkyIndex_std`: Variability in cloud cover within the hour

The statistical aggregations enhance the predictive power by providing information about the variance and extremes within each hour, which can significantly affect PV power generation.

## Recommendations for LSTM Model Training

1. **Sequence Length**: 24-48 hours recommended to capture daily patterns
2. **Feature Scaling**: Apply MinMaxScaler or StandardScaler to all features
3. **Feature Selection Refinement**: Consider backward elimination or sequential feature selection during model tuning
4. **Attention Mechanisms**: Consider LSTM with attention to focus on the most relevant time steps
5. **Hyperparameter Optimization**: Batch size, learning rate, and dropout rate should be tuned with cross-validation

## Usage in the LSTM Pipeline

```python
# Sample code for loading the enhanced feature set
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

# Load the prepared feature set with statistical aggregations
lstm_features = pd.read_parquet('data/lstm_features_with_aggregations.parquet')

# Split into features and target
X = lstm_features.drop(columns=['power_w'])
y = lstm_features['power_w']

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Now proceed with LSTM model preparation, sequence creation, etc.
```

## Conclusion

The enhanced feature set combines hourly predictors with statistical aggregations from higher resolution data, providing a more comprehensive view of the factors affecting PV power generation. This approach is expected to improve model performance by capturing important sub-hourly variations and transient weather events.