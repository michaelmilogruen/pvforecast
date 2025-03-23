# LSTM Low Resolution PV Power Forecasting

This project implements LSTM (Long Short-Term Memory) neural networks for forecasting photovoltaic (PV) power output using low-resolution (1-hour) weather and solar data.

## Overview

The implementation consists of several Python scripts:

1. `lstm_lowres.py` - Basic LSTM model implementation for 1-hour resolution data
2. `test_lstm_lowres.py` - Script to test the basic LSTM model
3. `plot_lstm_lowres_results.py` - Script to generate detailed analysis plots for the basic model
4. `lstm_lowres_improved.py` - Enhanced LSTM model with target scaling and more complex architecture
5. `test_lstm_lowres_improved.py` - Script to test the improved LSTM model

## Data

The model uses hourly weather and solar data from `data/station_data_1h.parquet`. This dataset includes:

- Weather measurements (temperature, global radiation, wind speed, pressure)
- Clear sky radiation components (GHI, DNI, DHI)
- Clear sky index
- Day/night indicator
- Temporal features (hour and day of year encoded as sine/cosine)
- PV power output (target variable)

## Basic LSTM Model (`lstm_lowres.py`)

The basic model provides a baseline implementation with:

- Feature scaling using appropriate scalers for different feature types
- Proper train/validation/test split
- Simple LSTM architecture
- Early stopping to prevent overfitting

### Usage

```bash
python src/lstm_lowres.py
```

This will:
1. Load and preprocess the data
2. Train the LSTM model
3. Save the trained model and scalers
4. Generate basic evaluation metrics

## Testing the Basic Model (`test_lstm_lowres.py`)

```bash
python src/test_lstm_lowres.py
```

This will:
1. Load the trained model and scalers
2. Evaluate the model on the test set
3. Generate predictions and performance metrics

## Detailed Analysis (`plot_lstm_lowres_results.py`)

```bash
python src/plot_lstm_lowres_results.py
```

This script generates detailed analysis plots:
- Scatter plot of actual vs. predicted values
- Time series comparison
- Residual plot
- Error distribution
- Performance by power range
- Hourly pattern analysis

## Improved LSTM Model (`lstm_lowres_improved.py`)

The improved model addresses limitations of the basic model:

- Target variable scaling to handle skewed power distribution
- More complex architecture with bidirectional LSTM layers
- Batch normalization for better training stability
- Longer sequence length (48 hours vs. 24 hours)
- Learning rate reduction on plateau

### Usage

```bash
python src/lstm_lowres_improved.py
```

## Testing the Improved Model (`test_lstm_lowres_improved.py`)

```bash
python src/test_lstm_lowres_improved.py
```

This will generate the same detailed analysis as the basic model test script, but for the improved model.

## Model Performance

### Basic Model
- RMSE: ~860 W
- MAE: ~310 W
- R²: ~0.16

### Improved Model
- RMSE: Significantly lower (expected)
- MAE: Significantly lower (expected)
- R²: Significantly higher (expected)

The improved model addresses the key limitations of the basic model, particularly the unscaled target variable and insufficient model complexity.

## Directory Structure

```
pvforecast/
├── data/
│   └── station_data_1h.parquet
├── models/
│   ├── lstm_lowres/
│   │   ├── final_model_[timestamp].keras
│   │   ├── minmax_scaler_[timestamp].pkl
│   │   ├── standard_scaler_[timestamp].pkl
│   │   ├── robust_scaler_[timestamp].pkl
│   │   └── target_scaler_[timestamp].pkl
│   └── lstm_lowres_improved/
│       ├── final_model_[timestamp].keras
│       ├── minmax_scaler_[timestamp].pkl
│       ├── standard_scaler_[timestamp].pkl
│       ├── robust_scaler_[timestamp].pkl
│       └── target_scaler_[timestamp].pkl
├── results/
│   ├── lstm_lowres_analysis/
│   │   ├── scatter_plot_[timestamp].png
│   │   ├── time_series_comparison_[timestamp].png
│   │   ├── residual_plot_[timestamp].png
│   │   ├── error_distribution_[timestamp].png
│   │   ├── rmse_by_power_range_[timestamp].png
│   │   ├── r2_by_power_range_[timestamp].png
│   │   ├── hourly_pattern_[timestamp].png
│   │   └── analysis_summary.md
│   └── lstm_lowres_improved_analysis/
│       ├── scatter_plot_[timestamp].png
│       ├── time_series_comparison_[timestamp].png
│       ├── residual_plot_[timestamp].png
│       ├── error_distribution_[timestamp].png
│       ├── rmse_by_power_range_[timestamp].png
│       ├── r2_by_power_range_[timestamp].png
│       └── hourly_pattern_[timestamp].png
└── src/
    ├── lstm_lowres.py
    ├── test_lstm_lowres.py
    ├── plot_lstm_lowres_results.py
    ├── lstm_lowres_improved.py
    └── test_lstm_lowres_improved.py
```

## Requirements

- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Joblib

## Future Improvements

1. Ensemble methods combining multiple models
2. Attention mechanisms for better temporal pattern recognition
3. Feature engineering to derive more predictive variables
4. Hyperparameter optimization using grid search or Bayesian optimization
5. Integration with weather forecast data for real-time predictions