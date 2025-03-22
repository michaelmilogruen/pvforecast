# PV Forecasting Data Analysis and Visualization

This directory contains visualizations generated from the analysis of photovoltaic (PV) power forecasting data. The visualizations are designed for scientific paper publication and provide comprehensive insights into the time series data characteristics.

## Overview

The visualizations are organized into several categories:

1. **Time Series Analysis**
   - Temporal patterns in solar radiation, temperature, and power output
   - Seasonal decomposition
   - Autocorrelation analysis
   - Trend analysis

2. **Statistical Distributions**
   - Distribution of key variables (radiation, temperature, wind speed, power)
   - Probability density functions
   - Quantile-quantile plots

3. **Correlation Analysis**
   - Correlation matrices
   - Feature importance for power prediction
   - Relationship between weather variables and power output

4. **Daily and Seasonal Patterns**
   - Diurnal patterns of radiation and power
   - Monthly/seasonal variations
   - Heatmaps of power output by hour and month

5. **Deep Learning Features**
   - Time series characteristics relevant for LSTM models
   - Sequence visualizations
   - Circular time encoding (sin/cos)
   - Normalized feature representations

6. **Temporal Resolution Comparison**
   - Analysis of 10-minute vs. 1-hour resolution data
   - Information loss in downsampling
   - Spectral analysis
   - Implications for model design

## Visualization Index

For easy navigation of all visualizations, open the `index.html` file in a web browser. This interactive index organizes all visualizations by category and provides a convenient way to browse the complete set of figures.

## Data Sources

The visualizations are based on two primary datasets:

1. `station_data_10min.parquet` - Weather station data at 10-minute resolution
2. `station_data_1h.parquet` - Weather station data at 1-hour resolution

These datasets contain the following key variables:
- Global radiation (W/m²)
- Temperature (°C)
- Wind speed (m/s)
- Clear sky index
- PV power output (W)
- Various derived features (clear sky radiation, time encodings, etc.)

## Generation Scripts

The visualizations were generated using the following Python scripts:

1. `src/analyze_station_data.py` - Basic station data analysis and visualization
2. `src/analyze_deep_learning_features.py` - Deep learning feature analysis
3. `src/analyze_temporal_resolution.py` - Temporal resolution comparison
4. `src/generate_all_visualizations.py` - Main script to run all analyses

To regenerate all visualizations, run:

```bash
python src/generate_all_visualizations.py
```

## Requirements

The analysis scripts require the following Python packages:
- pandas
- numpy
- matplotlib
- scipy
- statsmodels
- scikit-learn

## Citation

When using these visualizations in scientific publications, please cite the original data sources and analysis methodology.