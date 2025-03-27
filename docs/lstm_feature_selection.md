# LSTM Feature Selection for PV Power Prediction

## Overview

This document explains the feature selection process for the LSTM model that predicts `power_w` (PV power output). The selected features are saved in `data/lstm_features.parquet`.

## Feature Selection Methodology

The selection process incorporated:

1. **Correlation Analysis**: Pearson correlation coefficient with the target variable `power_w`
2. **Mutual Information Scores**: Capturing non-linear relationships
3. **Domain Knowledge**: Solar power generation physics and time-series patterns
4. **Redundancy Reduction**: Removing highly correlated features that provide duplicate information

## Selected Features

The final feature set includes:

| Feature | Selection Rationale | Correlation with power_w | MI Score |
|---------|---------------------|--------------------------|----------|
| GlobalRadiation [W m-2] | Primary driver of PV generation, highest correlation & MI score | 0.893 | 0.893 |
| ClearSkyGHI | Strong indicator of potential maximum irradiance | 0.827 | 0.759 |
| ClearSkyIndex | Ratio of actual to theoretical clear sky radiation, captures cloud effects | 0.651 | 0.557 |
| isNight | Binary feature indicating daylight availability | -0.633 | 0.482 |
| Temperature [degree_Celsius] | Affects panel efficiency (higher temp = lower efficiency) | 0.509 | 0.186 |
| WindSpeed [m s-1] | Affects panel cooling, moderately correlated | 0.377 | 0.126 |
| hour_sin, hour_cos | Captures diurnal (daily) patterns with cyclical representation | 0.124, -0.634 | 0.200, 0.566 |
| day_sin, day_cos | Captures seasonal patterns with cyclical representation | 0.020, -0.287 | 0.028, 0.070 |

## Features Excluded

Several features were excluded from the final selection:

| Excluded Feature | Reason for Exclusion |
|------------------|----------------------|
| ClearSkyDNI, ClearSkyDHI | Highly correlated with ClearSkyGHI (>0.9), redundant information |
| energy_wh, energy_interval | Direct derivatives of power_w, would create data leakage |
| Pressure [hPa] | Low correlation (-0.028) and MI score (0.022) with power_w |
| Precipitation [mm] | Low correlation (-0.048) and MI score (0.017) with power_w |
| hour, day_of_year | Replaced with sinusoidal representations for cyclical patterns |

## Data Preprocessing

- Missing values (702 across all features) were handled using forward-fill followed by backward-fill
- Original timestamp index was preserved for time-series prediction
- No scaling was applied in this feature selection phase (should be part of model training)

## Recommendations for LSTM Model Training

1. **Sequence Length**: Consider window sizes of 24-72 hours, covering full day cycles
2. **Feature Scaling**: Standardize or normalize all features to 0-1 range
3. **Validation Strategy**: Use time-based validation (not random) to prevent data leakage
4. **Model Architecture**:
   - Consider bidirectional LSTM for capturing future patterns
   - Use sufficient neurons (64-128) to capture complex relationships
   - Add dropout (0.2-0.3) to prevent overfitting
5. **Hyperparameter Tuning**: Focus on learning rate, batch size, and sequence length

## Usage in the LSTM Pipeline

```python
# Sample code for loading the feature set
import pandas as pd

# Load the prepared feature set
lstm_features = pd.read_parquet('data/lstm_features.parquet')

# Split into features and target
X = lstm_features.drop(columns=['power_w'])
y = lstm_features['power_w']

# Proceed with LSTM model preparation, scaling, etc.