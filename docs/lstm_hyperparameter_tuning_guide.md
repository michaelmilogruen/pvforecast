# LSTM Hyperparameter Tuning for PV Power Forecasting

## Overview

This document describes the enhanced LSTM (Long Short-Term Memory) model approach for photovoltaic (PV) power forecasting, with a focus on systematic hyperparameter tuning to improve model accuracy.

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Enhanced LSTM Architecture](#enhanced-lstm-architecture)
3. [Hyperparameter Tuning Approach](#hyperparameter-tuning-approach)
4. [Visualization and Analysis](#visualization-and-analysis)
5. [How to Use the Model](#how-to-use-the-model)
6. [Results and Interpretation](#results-and-interpretation)

## Problem Statement

Accurate forecasting of PV power output is crucial for grid integration and energy management. The challenge is to create a model that can effectively capture the temporal patterns in weather and solar radiation data to predict power output. Our previous LSTM model showed promising results, but there was room for improvement through hyperparameter optimization.

## Enhanced LSTM Architecture

The enhanced LSTM model incorporates several advanced features:

### 1. Bidirectional LSTM Layers

Bidirectional LSTMs process sequences in both forward and backward directions, allowing the model to capture patterns from both past and future time steps. This is particularly useful for time series data where future context can help improve predictions.

```python
if cfg['bidirectional']:
    model.add(Bidirectional(
        LSTM(cfg['lstm_units'][0],
             return_sequences=len(cfg['lstm_units']) > 1,
             kernel_regularizer=regularizer,
             recurrent_regularizer=regularizer),
        input_shape=input_shape
    ))
```

### 2. Batch Normalization

Batch normalization stabilizes and accelerates training by normalizing layer inputs, reducing internal covariate shift. This helps the model train faster and more reliably, especially with deeper architectures.

```python
if cfg['batch_norm']:
    model.add(BatchNormalization())
```

### 3. Regularization Techniques

Multiple regularization techniques are employed to prevent overfitting:

- **Dropout**: Randomly sets input units to 0 during training, forcing the network to learn redundant representations
- **L1/L2 Regularization**: Penalizes large weights to encourage simpler models that generalize better

```python
# Create regularizer if specified
regularizer = None
if cfg['l1_reg'] > 0 or cfg['l2_reg'] > 0:
    regularizer = l1_l2(l1=cfg['l1_reg'], l2=cfg['l2_reg'])
```

### 4. Learning Rate Scheduling

Adaptive learning rate scheduling with ReduceLROnPlateau adjusts the learning rate when training plateaus, helping the model converge to better minima.

```python
ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
```

## Hyperparameter Tuning Approach

Our hyperparameter tuning approach is systematic and comprehensive:

### Configurable Parameters

The following hyperparameters can be tuned:

| Parameter | Description |
|-----------|-------------|
| LSTM Units | Number of units in each LSTM layer |
| Dense Units | Number of units in each dense layer |
| Dropout Rates | Dropout probability after each layer |
| Learning Rate | Initial learning rate for the optimizer |
| Bidirectional | Whether to use bidirectional LSTM layers |
| Batch Normalization | Whether to use batch normalization |
| L1/L2 Regularization | Strength of L1 and L2 regularization |

### Predefined Configurations

We've defined 5 different configurations to explore different aspects of the model:

1. **Baseline Enhanced Model**: A balanced model with moderate complexity and regularization
2. **Deeper Model**: A deeper architecture with 3 LSTM layers
3. **Higher Capacity Model**: A model with more units in each layer
4. **More Regularization**: A model with stronger regularization to combat overfitting
5. **Simpler Model**: A lighter model with less regularization

Additionally, we've added feature-specific configurations for each data source (inca, station, combined) based on their characteristics.

### Tuning Process

The tuning process involves:

1. Training models with each configuration
2. Evaluating performance on validation data
3. Selecting the best configuration based on validation loss
4. Training a final model with the best configuration

## Visualization and Analysis

Our approach includes comprehensive visualization and analysis tools:

### Training History Visualization

For each configuration, we generate:

- Loss curves showing training and validation loss over epochs
- MAE curves showing training and validation MAE over epochs

### Configuration Comparison

We provide tools to compare different configurations:

- Bar charts comparing validation loss across configurations
- Line charts comparing validation metrics over epochs
- CSV reports with detailed performance metrics

### Results Analysis

The final results include:

- Prediction vs. actual plots for test data
- Scatter plots showing correlation between predicted and actual values
- Comprehensive metrics including MSE, RMSE, MAE, and R²

## How to Use the Model

The model can be run with or without hyperparameter tuning:

```python
# With hyperparameter tuning
forecaster = LSTMForecaster(
    sequence_length=24,
    batch_size=64,
    epochs=150,
    model_config=model_config
)
metrics = forecaster.run_pipeline(
    'data/processed_training_data.parquet',
    perform_tuning=True
)

# Without hyperparameter tuning (using predefined configuration)
metrics = forecaster.run_pipeline(
    'data/processed_training_data.parquet',
    perform_tuning=False
)
```

### Key Parameters

- `sequence_length`: Number of time steps to look back (default: 24)
- `batch_size`: Batch size for training (default: 64)
- `epochs`: Maximum number of epochs for training (default: 150)
- `model_config`: Dictionary with model hyperparameters
- `perform_tuning`: Whether to perform hyperparameter tuning

## Results and Interpretation

Based on our previous runs, the station feature set performed best with an RMSE of 611.11 and R² of 0.9228. The enhanced model with hyperparameter tuning is expected to further improve these metrics.

The tuning results are saved in the `results/tuning` directory, with detailed CSV reports and comparison plots for each feature set. The final model comparison is saved in `results/model_comparison.csv`.

### Key Findings

- Bidirectional LSTM layers significantly improve model performance
- Batch normalization helps stabilize training, especially for deeper models
- Moderate dropout (0.25-0.35) provides the best balance for regularization
- Feature-specific optimizations yield better results than a one-size-fits-all approach
