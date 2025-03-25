#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LSTM Model for PV Power Forecasting with Low Resolution (1-hour) Data

This script implements an LSTM (Long Short-Term Memory) neural network for forecasting 
photovoltaic power output using 1-hour resolution data. It includes data loading, 
preprocessing, model training, and evaluation.

Key features:
1. Data preprocessing and feature scaling
2. LSTM model architecture
3. Hyperparameter optimization using Bayesian optimization with Optuna
4. Model evaluation and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import optuna
from datetime import datetime
import argparse

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class LSTMLowResForecaster:
    def __init__(self, sequence_length=24, batch_size=32, epochs=50):
        """
        Initialize the LSTM forecaster for low-resolution data.
        
        Args:
            sequence_length: Number of time steps to look back (default: 24 hours)
            batch_size: Batch size for training
            epochs: Maximum number of epochs for training
        """
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Create directories for model and results
        os.makedirs('models/lstm_lowres', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        os.makedirs('results/lstm_lowres_optuna', exist_ok=True)
        
        # Timestamp for model versioning
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Default model configuration (will be updated by hyperparameter optimization)
        self.config = {
            'lstm_units': [64, 32, 16],  # Default 3 LSTM layers
            'dense_units': [16, 8],      # Default 2 dense layers
            'dropout_rates': [0.2, 0.15, 0.1],  # Adaptive dropout rates for LSTM layers
            'dense_dropout_rates': [0.1, 0.05],  # Adaptive dropout rates for dense layers
            'learning_rate': 0.001,
            'bidirectional': False,
            'batch_norm': True  # Default to True to allow optimization to try without it
        }
    
    def load_and_prepare_data(self, data_path):
        """
        Load and prepare data for training.
        
        Args:
            data_path: Path to the parquet file containing the data
            
        Returns:
            Processed dataframe
        """
        print(f"Loading data from {data_path}...")
        df = pd.read_parquet(data_path)
        print(f"Data shape: {df.shape}")
        print(f"Data range: {df.index.min()} to {df.index.max()}")
        
        # Check for missing values
        missing_values = df.isna().sum()
        if missing_values.sum() > 0:
            print("Missing values in dataset:")
            print(missing_values[missing_values > 0])
            
            # Fill missing values
            print("Filling missing values...")
            # For numeric columns, use forward fill then backward fill
            df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def prepare_features(self, df):
        """
        Prepare features for the model.
        
        Args:
            df: DataFrame with all features
            
        Returns:
            Dictionary with feature information
        """
        # Define features to use
        features = [
            'GlobalRadiation [W m-2]',
            'Temperature [degree_Celsius]',
            'WindSpeed [m s-1]',
            'ClearSkyIndex',
            'hour_sin',
            'hour_cos',
            'day_sin',
            'day_cos',
            'isNight'
        ]
        
        # Verify all features exist in the dataframe
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            # Remove missing features
            features = [f for f in features if f in df.columns]
        
        # Group features by scaling method based on their distributions
        # Based on the distribution plots:
        # - Global Radiation and ClearSkyIndex: Highly skewed - MinMaxScaler
        # - Temperature: More normally distributed - StandardScaler
        # - Wind Speed: Right-skewed with outliers - RobustScaler
        # - Time features (sin/cos) and isNight: Already normalized - No scaling needed
        
        minmax_features = ['GlobalRadiation [W m-2]', 'ClearSkyIndex']
        standard_features = ['Temperature [degree_Celsius]']
        robust_features = ['WindSpeed [m s-1]']
        no_scaling_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'isNight']
        
        # Filter to only include features that exist in the dataframe
        minmax_features = [f for f in minmax_features if f in features]
        standard_features = [f for f in standard_features if f in features]
        robust_features = [f for f in robust_features if f in features]
        no_scaling_features = [f for f in no_scaling_features if f in features]
        
        print(f"MinMaxScaler features: {minmax_features}")
        print(f"StandardScaler features: {standard_features}")
        print(f"RobustScaler features: {robust_features}")
        print(f"No scaling features: {no_scaling_features}")
        
        target = 'power_w'
        
        return {
            'all_features': features,
            'minmax_features': minmax_features,
            'standard_features': standard_features,
            'robust_features': robust_features,
            'no_scaling_features': no_scaling_features,
            'target': target
        }
    
    def split_and_scale_data(self, df, feature_info):
        """
        Split data into train, validation, and test sets, then scale features.
        
        Args:
            df: DataFrame with all features
            feature_info: Dictionary with feature information
            
        Returns:
            Dictionary with split and scaled data
        """
        # Split data into train, validation, and test sets
        # Use 70% for training, 15% for validation, 15% for testing
        train_size = int(len(df) * 0.7)
        val_size = int(len(df) * 0.15)
        
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:train_size+val_size]
        test_df = df.iloc[train_size+val_size:]
        
        print(f"Train set size: {len(train_df)}")
        print(f"Validation set size: {len(val_df)}")
        print(f"Test set size: {len(test_df)}")
        
        # Initialize scalers for different feature groups
        minmax_scaler = MinMaxScaler() if feature_info['minmax_features'] else None
        standard_scaler = StandardScaler() if feature_info['standard_features'] else None
        robust_scaler = RobustScaler() if feature_info['robust_features'] else None
        
        # Initialize target scaler
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Fit scalers on training data only to prevent data leakage
        if minmax_scaler:
            minmax_scaler.fit(train_df[feature_info['minmax_features']])
            joblib.dump(minmax_scaler, f'models/lstm_lowres/minmax_scaler_{self.timestamp}.pkl')
        
        if standard_scaler:
            standard_scaler.fit(train_df[feature_info['standard_features']])
            joblib.dump(standard_scaler, f'models/lstm_lowres/standard_scaler_{self.timestamp}.pkl')
        
        if robust_scaler:
            robust_scaler.fit(train_df[feature_info['robust_features']])
            joblib.dump(robust_scaler, f'models/lstm_lowres/robust_scaler_{self.timestamp}.pkl')
        
        # Fit target scaler on training data
        target_scaler.fit(train_df[[feature_info['target']]])
        joblib.dump(target_scaler, f'models/lstm_lowres/target_scaler_{self.timestamp}.pkl')
        
        # Function to scale features using appropriate scalers
        def scale_features(df_subset):
            scaled_data = {}
            
            # Apply MinMaxScaler to appropriate features
            if minmax_scaler and feature_info['minmax_features']:
                scaled_minmax = minmax_scaler.transform(df_subset[feature_info['minmax_features']])
                for i, feature in enumerate(feature_info['minmax_features']):
                    scaled_data[feature] = scaled_minmax[:, i]
            
            # Apply StandardScaler to appropriate features
            if standard_scaler and feature_info['standard_features']:
                scaled_standard = standard_scaler.transform(df_subset[feature_info['standard_features']])
                for i, feature in enumerate(feature_info['standard_features']):
                    scaled_data[feature] = scaled_standard[:, i]
            
            # Apply RobustScaler to appropriate features
            if robust_scaler and feature_info['robust_features']:
                scaled_robust = robust_scaler.transform(df_subset[feature_info['robust_features']])
                for i, feature in enumerate(feature_info['robust_features']):
                    scaled_data[feature] = scaled_robust[:, i]
            
            # Add features that don't need scaling
            for feature in feature_info['no_scaling_features']:
                scaled_data[feature] = df_subset[feature].values
            
            # Create DataFrame with all scaled features
            scaled_df = pd.DataFrame(scaled_data, index=df_subset.index)
            
            # Ensure columns are in the same order as all_features
            return scaled_df[feature_info['all_features']]
        
        # Apply scaling to each dataset
        X_train = scale_features(train_df)
        X_val = scale_features(val_df)
        X_test = scale_features(test_df)
        
        # Scale target variable
        y_train = target_scaler.transform(train_df[[feature_info['target']]])
        y_val = target_scaler.transform(val_df[[feature_info['target']]])
        y_test = target_scaler.transform(test_df[[feature_info['target']]])
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'scalers': {
                'minmax': minmax_scaler,
                'standard': standard_scaler,
                'robust': robust_scaler,
                'target': target_scaler
            }
        }
    
    def create_sequences(self, X, y, time_steps):
        """
        Create sequences for time series forecasting.
        
        Args:
            X: Features DataFrame
            y: Target array
            time_steps: Number of time steps to look back
            
        Returns:
            X_seq: Sequences of features
            y_seq: Corresponding target values
        """
        X_seq, y_seq = [], []
        
        for i in range(len(X) - time_steps):
            X_seq.append(X.iloc[i:i + time_steps].values)
            y_seq.append(y[i + time_steps])
            
        return np.array(X_seq), np.array(y_seq)
    
    def build_model(self, input_shape, trial=None):
        """
        Build an LSTM model with optional hyperparameters from Optuna trial.
        
        Args:
            input_shape: Shape of input data (time_steps, features)
            trial: Optuna trial object for hyperparameter optimization (optional)
            
        Returns:
            Compiled LSTM model
        """
        model = Sequential()
        
        # If trial is provided, use it to suggest hyperparameters
        if trial:
            # Suggest hyperparameters using Optuna
            # Number of layers
            num_lstm_layers = trial.suggest_int('num_lstm_layers', 1, 3)
            num_dense_layers = trial.suggest_int('num_dense_layers', 1, 3)
            
            # LSTM units for each layer - use consistent parameter ranges for each position
            lstm_units = []
            if num_lstm_layers >= 1:
                # First layer typically has more units
                lstm_units.append(trial.suggest_int('lstm_units_1', 32, 256))
            
            if num_lstm_layers >= 2:
                # Second layer
                lstm_units.append(trial.suggest_int('lstm_units_2', 16, 128))
            
            if num_lstm_layers >= 3:
                # Third layer typically has fewer units
                lstm_units.append(trial.suggest_int('lstm_units_3', 8, 64))
            
            # Dense units for each layer - use consistent parameter ranges for each position
            dense_units = []
            if num_dense_layers >= 1:
                # First dense layer
                dense_units.append(trial.suggest_int('dense_units_1', 16, 64))
            
            if num_dense_layers >= 2:
                # Second dense layer
                dense_units.append(trial.suggest_int('dense_units_2', 8, 32))
                
            if num_dense_layers >= 3:
                # Third dense layer
                dense_units.append(trial.suggest_int('dense_units_3', 4, 16))
            # Suggest different dropout rates for each layer
            lstm_dropout_rates = []
            for i in range(1, num_lstm_layers + 1):
                lstm_dropout_rates.append(trial.suggest_float(f'lstm_dropout_rate_{i}', 0.1, 0.5))
            
            dense_dropout_rates = []
            for i in range(1, num_dense_layers):  # No dropout after the last dense layer
                dense_dropout_rates.append(trial.suggest_float(f'dense_dropout_rate_{i}', 0.05, 0.3))
                
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            bidirectional = trial.suggest_categorical('bidirectional', [True, False])
            batch_norm = trial.suggest_categorical('batch_norm', [True])
            
            # Update config with suggested hyperparameters
            dropout_rates = lstm_dropout_rates
            dense_dropout_rates = dense_dropout_rates
        else:
            # Use existing configuration
            lstm_units = self.config['lstm_units']
            dense_units = self.config['dense_units']
            dropout_rates = self.config['dropout_rates']
            dense_dropout_rates = self.config['dense_dropout_rates']
            learning_rate = self.config['learning_rate']
            bidirectional = self.config['bidirectional']
            batch_norm = self.config['batch_norm']
            num_lstm_layers = len(lstm_units)
            num_dense_layers = len(dense_units)
        
        # First LSTM layer
        if bidirectional:
            model.add(Bidirectional(
                LSTM(units=lstm_units[0], return_sequences=(num_lstm_layers > 1)),
                input_shape=input_shape
            ))
        else:
            model.add(LSTM(
                units=lstm_units[0],
                return_sequences=(num_lstm_layers > 1),
                input_shape=input_shape
            ))
        
        if batch_norm:
            model.add(BatchNormalization())
        
        model.add(Dropout(dropout_rates[0]))
        
        # Middle LSTM layers (if any)
        for i in range(1, num_lstm_layers):
            is_last_lstm = (i == num_lstm_layers - 1)
            if bidirectional:
                model.add(Bidirectional(
                    LSTM(units=lstm_units[i], return_sequences=not is_last_lstm)
                ))
            else:
                model.add(LSTM(
                    units=lstm_units[i],
                    return_sequences=not is_last_lstm
                ))
            
            if batch_norm:
                model.add(BatchNormalization())
            
            model.add(Dropout(dropout_rates[i] if i < len(dropout_rates) else dropout_rates[-1]))  # Use appropriate LSTM layer dropout rate
        
        # Dense layers
        for i in range(num_dense_layers):
            model.add(Dense(dense_units[i], activation='relu'))
            
            if batch_norm:
                model.add(BatchNormalization())
            
            if i < num_dense_layers - 1:  # No dropout after the last dense layer
                model.add(Dropout(dense_dropout_rates[i] if i < len(dense_dropout_rates) else dense_dropout_rates[-1]))  # Use appropriate dense layer dropout rate
        
        # Output layer
        model.add(Dense(1))
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def create_callbacks(self, trial=None):
        """
        Create callbacks for model training.
        
        Args:
            trial: Optuna trial object (optional)
            
        Returns:
            List of callbacks
        """
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
        
        # Model checkpoint (only for final model, not during hyperparameter search)
        if trial is None:
            model_checkpoint = ModelCheckpoint(
                filepath=f'models/lstm_lowres/model_{self.timestamp}.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
            callbacks.append(model_checkpoint)
        
        # Learning rate reduction
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0 if trial else 1  # Less verbose during hyperparameter search
        )
        callbacks.append(reduce_lr)
        
        # Add Optuna pruning callback if trial is provided
        if trial:
            pruning_callback = optuna.integration.TFKerasPruningCallback(
                trial, 'val_loss'
            )
            callbacks.append(pruning_callback)
        
        return callbacks
    
    def objective(self, trial, X_train_seq, y_train_seq, X_val_seq, y_val_seq):
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            X_train_seq: Training sequences
            y_train_seq: Training targets
            X_val_seq: Validation sequences
            y_val_seq: Validation targets
            
        Returns:
            Validation loss (to be minimized)
        """
        # Build model with hyperparameters from trial
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        model = self.build_model(input_shape, trial)
        
        # Create callbacks with pruning
        callbacks = self.create_callbacks(trial)
        
        # Train model with fewer epochs for hyperparameter search
        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=20,  # Reduced epochs for hyperparameter search
            batch_size=self.batch_size,
            validation_data=(X_val_seq, y_val_seq),
            callbacks=callbacks,
            verbose=0  # Silent training during hyperparameter search
        )
        
        # Return the best validation loss
        return history.history['val_loss'][-1]
    
    def optimize_hyperparameters(self, X_train_seq, y_train_seq, X_val_seq, y_val_seq, n_trials=50):
        """
        Optimize hyperparameters using Optuna's Bayesian optimization.
        
        Args:
            X_train_seq: Training sequences
            y_train_seq: Training targets
            X_val_seq: Validation sequences
            y_val_seq: Validation targets
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary with optimized hyperparameters
        """
        print("\nStarting hyperparameter optimization using Bayesian optimization with Optuna...")
        
        # Create a study object with pruning
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
            interval_steps=1
        )
        
        study = optuna.create_study(
            direction='minimize',
            pruner=pruner,
            study_name=f'lstm_lowres_study_{self.timestamp}'
        )
        
        # Define the objective function
        objective_func = lambda trial: self.objective(
            trial, X_train_seq, y_train_seq, X_val_seq, y_val_seq
        )
        
        # Run the optimization
        study.optimize(objective_func, n_trials=n_trials)
        
        # Get best parameters
        best_params = study.best_params
        print("\nBest hyperparameters:")
        for param, value in best_params.items():
            print(f"{param}: {value}")
        
        # Update config with best parameters
        num_lstm_layers = best_params['num_lstm_layers']
        num_dense_layers = best_params['num_dense_layers']
        
        # Extract LSTM units
        lstm_units = []
        for i in range(1, num_lstm_layers + 1):
            if f'lstm_units_{i}' in best_params:
                lstm_units.append(best_params[f'lstm_units_{i}'])
        self.config['lstm_units'] = lstm_units
        
        # Extract dense units
        dense_units = []
        for i in range(1, num_dense_layers + 1):
            if f'dense_units_{i}' in best_params:
                dense_units.append(best_params[f'dense_units_{i}'])
        self.config['dense_units'] = dense_units
        # Extract adaptive dropout rates for LSTM and dense layers
        lstm_dropout_rates = []
        for i in range(1, num_lstm_layers + 1):
            if f'lstm_dropout_rate_{i}' in best_params:
                lstm_dropout_rates.append(best_params[f'lstm_dropout_rate_{i}'])
        self.config['dropout_rates'] = lstm_dropout_rates
        
        # Extract dense dropout rates
        dense_dropout_rates = []
        for i in range(1, num_dense_layers):  # No dropout after last dense layer
            if f'dense_dropout_rate_{i}' in best_params:
                dense_dropout_rates.append(best_params[f'dense_dropout_rate_{i}'])
        self.config['dense_dropout_rates'] = dense_dropout_rates
        self.config['learning_rate'] = best_params['learning_rate']
        self.config['bidirectional'] = best_params['bidirectional']
        self.config['batch_norm'] = best_params['batch_norm']
        
        # Save hyperparameter search results
        self.plot_optuna_results(study)
        
        # Save study for later analysis
        joblib.dump(study, f'results/lstm_lowres_optuna/study_{self.timestamp}.pkl')
        
        return best_params
    
    def plot_optuna_results(self, study):
        """
        Plot and save Optuna optimization results.
        
        Args:
            study: Optuna study object
        """
        # Create directory for plots
        os.makedirs('results/lstm_lowres_optuna', exist_ok=True)
        
        # Plot optimization history
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig(f'results/lstm_lowres_optuna/optimization_history_{self.timestamp}.png')
        
        # Plot parameter importances
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.tight_layout()
        plt.savefig(f'results/lstm_lowres_optuna/param_importances_{self.timestamp}.png')
        
        # Plot parallel coordinate plot
        plt.figure(figsize=(12, 8))
        optuna.visualization.matplotlib.plot_parallel_coordinate(study)
        plt.tight_layout()
        plt.savefig(f'results/lstm_lowres_optuna/parallel_coordinate_{self.timestamp}.png')
        
        # Plot slice plots for key hyperparameters that exist in the study
        key_params = ['num_lstm_layers', 'num_dense_layers', 'learning_rate']
        # Get all parameter names from the study
        study_params = list(study.best_params.keys())
        
        # Plot slice plots for primary parameters
        for param in key_params:
            if param in study_params:
                plt.figure(figsize=(10, 6))
                optuna.visualization.matplotlib.plot_slice(study, params=[param])
                plt.tight_layout()
                plt.savefig(f'results/lstm_lowres_optuna/slice_{param}_{self.timestamp}.png')
        
        # Plot slice plots for dropout rates if they exist
        dropout_params = [p for p in study_params if 'dropout_rate' in p]
        for param in dropout_params:
            plt.figure(figsize=(10, 6))
            optuna.visualization.matplotlib.plot_slice(study, params=[param])
            plt.tight_layout()
            plt.savefig(f'results/lstm_lowres_optuna/slice_{param}_{self.timestamp}.png')
        
        print(f"Optuna visualization plots saved to results/lstm_lowres_optuna/")
    
    def evaluate_model(self, model, X_test_seq, y_test, target_scaler=None):
        """
        Evaluate the model and return metrics.
        
        Args:
            model: Trained LSTM model
            X_test_seq: Test sequences
            y_test: True target values
            target_scaler: Scaler for the target variable (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test_seq)
        
        # Calculate metrics on scaled data
        mse_scaled = mean_squared_error(y_test, y_pred)
        rmse_scaled = np.sqrt(mse_scaled)
        mae_scaled = mean_absolute_error(y_test, y_pred)
        r2_scaled = r2_score(y_test, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error) for scaled data
        # Avoid division by zero or very small numbers
        epsilon = 1e-10  # Small constant to avoid division by zero
        mape_scaled = np.mean(np.abs((y_test - y_pred) / (y_test + epsilon))) * 100
        
        # Calculate SMAPE (Symmetric Mean Absolute Percentage Error) for scaled data
        # This metric is symmetric and handles zero values better
        smape_scaled = np.mean(2.0 * np.abs(y_pred - y_test) / (np.abs(y_pred) + np.abs(y_test) + epsilon)) * 100
        
        print("\nModel Evaluation (Scaled):")
        print(f"Mean Squared Error (MSE): {mse_scaled:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse_scaled:.4f}")
        print(f"Mean Absolute Error (MAE): {mae_scaled:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape_scaled:.2f}%")
        print(f"Symmetric Mean Absolute Percentage Error (SMAPE): {smape_scaled:.2f}%")
        print(f"R² Score: {r2_scaled:.4f}")
        
        # If target scaler is provided, inverse transform predictions and true values
        if target_scaler:
            y_test_inv = target_scaler.inverse_transform(y_test)
            y_pred_inv = target_scaler.inverse_transform(y_pred)
            
            # Calculate metrics on original scale
            mse = mean_squared_error(y_test_inv, y_pred_inv)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_inv, y_pred_inv)
            r2 = r2_score(y_test_inv, y_pred_inv)
            
            # Calculate MAPE and SMAPE on original scale
            epsilon = 1e-10  # Small constant to avoid division by zero
            mape = np.mean(np.abs((y_test_inv - y_pred_inv) / (y_test_inv + epsilon))) * 100
            smape = np.mean(2.0 * np.abs(y_pred_inv - y_test_inv) / (np.abs(y_pred_inv) + np.abs(y_test_inv) + epsilon)) * 100
            
            print("\nModel Evaluation (Original Scale):")
            print(f"Mean Squared Error (MSE): {mse:.4f}")
            print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
            print(f"Mean Absolute Error (MAE): {mae:.4f}")
            print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
            print(f"Symmetric Mean Absolute Percentage Error (SMAPE): {smape:.2f}%")
            print(f"R² Score: {r2:.4f}")
            
            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'smape': smape,
                'r2': r2,
                'mse_scaled': mse_scaled,
                'rmse_scaled': rmse_scaled,
                'mae_scaled': mae_scaled,
                'mape_scaled': mape_scaled,
                'smape_scaled': smape_scaled,
                'r2_scaled': r2_scaled,
                'y_test_inv': y_test_inv,
                'y_pred_inv': y_pred_inv,
                'y_test': y_test,
                'y_pred': y_pred
            }
        else:
            # No target scaling was applied
            return {
                'mse': mse_scaled,
                'rmse': rmse_scaled,
                'mae': mae_scaled,
                'mape': mape_scaled,
                'smape': smape_scaled,
                'r2': r2_scaled,
                'y_test': y_test,
                'y_pred': y_pred
            }
    
    def plot_results(self, history, evaluation_results):
        """
        Plot and save training history and prediction results.
        
        Args:
            history: Training history
            evaluation_results: Results from model evaluation
        """
        # Plot training history
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'results/lstm_lowres_history_{self.timestamp}.png')
        
        # Check if we have inverse-transformed data
        has_inverse_transform = 'y_test_inv' in evaluation_results
        
        # Plot predictions vs actual (scaled)
        plt.figure(figsize=(14, 7))
        
        sample_size = min(500, len(evaluation_results['y_test']))
        indices = np.arange(sample_size)
        
        plt.plot(indices, evaluation_results['y_test'][:sample_size], 'b-', label='Actual (Scaled)')
        plt.plot(indices, evaluation_results['y_pred'][:sample_size], 'r-', label='Predicted (Scaled)')
        plt.title('Actual vs Predicted PV Power (Scaled)')
        plt.xlabel('Time Steps')
        plt.ylabel('Power Output (Scaled)')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(f'results/lstm_lowres_predictions_scaled_{self.timestamp}.png')
        
        # If we have inverse-transformed data, plot it
        if has_inverse_transform:
            # Plot predictions vs actual (original scale)
            plt.figure(figsize=(14, 7))
            
            plt.plot(indices, evaluation_results['y_test_inv'][:sample_size], 'b-', label='Actual Power Output')
            plt.plot(indices, evaluation_results['y_pred_inv'][:sample_size], 'r-', label='Predicted Power Output')
            plt.title('Actual vs Predicted PV Power')
            plt.xlabel('Time Steps')
            plt.ylabel('Power Output (W)')
            plt.legend()
            plt.grid(True)
            
            plt.savefig(f'results/lstm_lowres_predictions_{self.timestamp}.png')
            
            # Create scatter plot (original scale)
            plt.figure(figsize=(10, 8))
            plt.scatter(evaluation_results['y_test_inv'], evaluation_results['y_pred_inv'], alpha=0.5)
            plt.plot([evaluation_results['y_test_inv'].min(), evaluation_results['y_test_inv'].max()], 
                     [evaluation_results['y_test_inv'].min(), evaluation_results['y_test_inv'].max()], 'r--')
            plt.title('Actual vs Predicted')
            plt.xlabel('Actual Power Output (W)')
            plt.ylabel('Predicted Power Output (W)')
            plt.grid(True)
            
            plt.savefig(f'results/lstm_lowres_scatter_{self.timestamp}.png')
        else:
            # Create scatter plot (scaled)
            plt.figure(figsize=(10, 8))
            plt.scatter(evaluation_results['y_test'], evaluation_results['y_pred'], alpha=0.5)
            plt.plot([evaluation_results['y_test'].min(), evaluation_results['y_test'].max()], 
                     [evaluation_results['y_test'].min(), evaluation_results['y_test'].max()], 'r--')
            plt.title('Actual vs Predicted (Scaled)')
            plt.xlabel('Actual Power Output (Scaled)')
            plt.ylabel('Predicted Power Output (Scaled)')
            plt.grid(True)
            
            plt.savefig(f'results/lstm_lowres_scatter_{self.timestamp}.png')
    
    def run_pipeline(self, optimize_hyperparams=True, n_trials=50):
        """
        Run the full LSTM forecasting pipeline for low-resolution data.
        
        Args:
            optimize_hyperparams: Whether to perform hyperparameter optimization
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("Running LSTM forecasting pipeline for low-resolution (1-hour) data")
        
        # Load and prepare data
        df = self.load_and_prepare_data('data/station_data_1h.parquet')
        
        # Prepare features
        feature_info = self.prepare_features(df)
        
        # Split and scale data
        data = self.split_and_scale_data(df, feature_info)
        
        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(
            data['X_train'], data['y_train'], self.sequence_length
        )
        X_val_seq, y_val_seq = self.create_sequences(
            data['X_val'], data['y_val'], self.sequence_length
        )
        X_test_seq, y_test_seq = self.create_sequences(
            data['X_test'], data['y_test'], self.sequence_length
        )
        
        print(f"Training sequences shape: {X_train_seq.shape}")
        print(f"Validation sequences shape: {X_val_seq.shape}")
        print(f"Testing sequences shape: {X_test_seq.shape}")
        
        # Perform hyperparameter optimization if requested
        if optimize_hyperparams:
            print("\nPerforming hyperparameter optimization...")
            best_params = self.optimize_hyperparameters(
                X_train_seq, y_train_seq, X_val_seq, y_val_seq, n_trials=n_trials
            )
            print(f"Optimized hyperparameters: {best_params}")
            # Save optimized hyperparameters
            with open(f'models/lstm_lowres/best_params_{self.timestamp}.txt', 'w', encoding='utf-8') as f:
                for param, value in best_params.items():
                    f.write(f"{param}: {value}\n")
                    f.write(f"{param}: {value}\n")
        
        # Build model with optimized hyperparameters
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        model = self.build_model(input_shape)
        model.summary()
        
        # Create callbacks
        callbacks = self.create_callbacks()
        
        # Train model
        print("\nTraining model...")
        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val_seq, y_val_seq),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        print("\nEvaluating model...")
        evaluation_results = self.evaluate_model(
            model, X_test_seq, y_test_seq, data['scalers']['target']
        )
        
        # Plot results
        self.plot_results(history, evaluation_results)
        
        # Save final model
        model.save(f'models/lstm_lowres/final_model_{self.timestamp}.keras')
        print(f"Model saved to models/lstm_lowres/final_model_{self.timestamp}.keras")
        
        # Save model summary to file
        with open(f'models/lstm_lowres/model_summary_{self.timestamp}.txt', 'w', encoding='utf-8') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        return evaluation_results


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run LSTM model with Bayesian hyperparameter optimization')
    parser.add_argument('--optimize', action='store_true', help='Perform hyperparameter optimization')
    parser.add_argument('--sequence_length', type=int, default=24, help='Sequence length (hours to look back)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Maximum number of epochs for training')
    parser.add_argument('--trials', type=int, default=50, help='Number of optimization trials')
    args = parser.parse_args()
    
    # Check for GPU availability
    print("TensorFlow version:", tf.__version__)
    print("GPU Available:", tf.config.list_physical_devices('GPU'))
    
    # Set memory growth to avoid memory allocation issues
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory growth set to True for all GPUs")
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")
    
    # Set parameters
    SEQUENCE_LENGTH = args.sequence_length
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    OPTIMIZE = args.optimize
    N_TRIALS = args.trials
    
    print(f"Running LSTM model with parameters:")
    print(f"- Sequence length: {SEQUENCE_LENGTH}")
    print(f"- Batch size: {BATCH_SIZE}")
    print(f"- Max epochs: {EPOCHS}")
    print(f"- Hyperparameter optimization: {'Enabled' if OPTIMIZE else 'Disabled'}")
    if OPTIMIZE:
        print(f"- Number of optimization trials: {N_TRIALS}")
    
    # Initialize and run forecaster
    forecaster = LSTMLowResForecaster(
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )
    
    # Run pipeline
    metrics = forecaster.run_pipeline(optimize_hyperparams=OPTIMIZE, n_trials=N_TRIALS)
    
    # Print final results
    print("\nFinal Results:")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"SMAPE: {metrics['smape']:.2f}%")
    print(f"R²: {metrics['r2']:.4f}")
    if 'r2_scaled' in metrics:
        print(f"R² (Scaled): {metrics['r2_scaled']:.4f}")