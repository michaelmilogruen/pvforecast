#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LSTM Model for PV Power Forecasting with Low Resolution (1-hour) Data

This script implements an LSTM (Long Short-Term Memory) neural network for forecasting
photovoltaic power output using 1-hour resolution data. It includes data loading,
preprocessing, time series cross-validation for tuning (optional), and either TSCV
for final training (if n_splits_tscv > 1 is explicitly specified) or single model
training (if n_splits_tscv is not specified or <= 1), followed by evaluation on a
separate, untouched test set.

Key features:
1. Data preprocessing and feature scaling. Scalers are fitted only on the training
   portion of the data to prevent leakage. The target variable is also scaled.
2. Time Series Cross-Validation (TSCV) for robust hyperparameter tuning using Optuna (optional).
3. LSTM model architecture.
4. Conditional implementation of TSCV for the final model training phase based on
   the 'n_splits_tscv' argument. If n_splits_tscv > 1, multiple models are trained
   and predictions are averaged. If n_splits_tscv <= 1, a single model is trained.
5. Averaging of predictions (if TSCV for final training is used) or direct prediction
   (if single model training) for final evaluation on the separate test set.
6. Model evaluation on a separate, final test set. **Metrics on the original scale
   (MSE, RMSE, MAE, MAPE, SMAPE) are now correctly calculated using inverse transformed values.**
7. Visualization of training history (from the last fold or single training),
   prediction results (using averaged or single predictions), and TSCV splits (if optimization is on).
8. Modified to load data from a specified parquet file and use a specific list
   of features, including 'hour_cos' and 'isNight', without additional data augmentation.
9. Added function to plot actual vs predicted power for the entire test set.
10. Modified plotting functions to keep plots open after saving.
11. Added visualization for Time Series Cross-Validation splits.
12. Fixed KeyError by using a class attribute for the radiation column name.
13. Corrected logic to perform final training TSCV only if n_splits_tscv > 1 is explicitly parsed.
14. Fixed indentation issues.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Lambda
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit # Import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import optuna
from datetime import datetime
import argparse
import math # Import math for pi
from matplotlib.patches import Patch # Import Patch for legend

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define the clipping function outside the class/method scope
# This ensures 'tf' is accessible when the function is called by the Lambda layer
def clip_scaled_output(x):
    """Clips the scaled output to be non-negative."""
    return tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=tf.float32.max)


class LSTMLowResForecaster:
    def __init__(self, sequence_length=3, batch_size=16, epochs=50):
        """
        Initialize the LSTM forecaster for low-resolution data.

        Args:
            sequence_length: Number of time steps to look back (default: 3 hours)
            batch_size: Batch size for training
            epochs: Maximum number of epochs for training
        """
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.epochs = epochs

        # Create directories for model and results
        self.model_dir = 'models/lstm_lowres'
        self.results_dir = 'results'
        self.optuna_results_dir = os.path.join(self.results_dir, 'lstm_lowres_optuna')
        # Directory for storing models from final TSCV training folds
        self.final_models_dir = os.path.join(self.model_dir, 'final_tscv_models')
        os.makedirs(self.final_models_dir, exist_ok=True)


        # Ensure directories exist
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.optuna_results_dir, exist_ok=True)

        # Timestamp for model versioning
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Name of the global radiation column (used for isNight and post-processing)
        self.radiation_column = 'GlobalRadiation [W m-2]'

        # Default model configuration (will be updated by hyperparameter optimization)
        self.config = {
            'lstm_units': [32],  # Default 3 LSTM layers
            'dense_units': [16],      # Default 2 dense layers
            'dropout_rates': [0.3],  # Adaptive dropout rates for LSTM layers
            'dense_dropout_rates': [0.1],  # Adaptive dropout rates for dense layers
            'learning_rate': 0.001,
            'bidirectional': False,
            'batch_norm': True    # Default to True to allow optimization to try without it
        }

    def load_data(self, data_path):
        """
        Load data for training from the specified file path.

        Args:
            data_path: Path to the parquet file containing the data.

        Returns:
            Loaded dataframe with datetime index.
        """
        print(f"Loading data from {data_path}...")
        try:
            # Reading the parquet file as specified
            df = pd.read_parquet(data_path)
            print(f"Data shape: {df.shape}")

            # Ensure the index is a datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                print("Warning: DataFrame index is not DatetimeIndex. Attempting to convert.")
                try:
                    # Assuming the index column is named 'index' or similar, or is the first column
                    # You might need to adjust this based on your parquet file structure
                    if 'index' in df.columns:
                        df['index'] = pd.to_datetime(df['index'])
                        df = df.set_index('index')
                    else:
                        # Attempt to use the current index if it's not DatetimeIndex
                        df.index = pd.to_datetime(df.index)
                    print("Index converted to DatetimeIndex.")
                except Exception as e_index:
                    print(f"Error converting index to datetime: {e_index}")
                    print("Please ensure your data file has a proper datetime index or modify the load_data method.")
                    exit()

            print(f"Data range: {df.index.min()} to {df.index.max()}")

            # Check for missing values
            missing_values = df.isna().sum()
            if missing_values.sum() > 0:
                print("Missing values in dataset:")
                print(missing_values[missing_values > 0])

                # Fill missing values (standard data cleaning, not augmentation)
                print("Filling missing values using ffill followed by bfill...")
                # For numeric columns, use forward fill then backward fill
                df = df.fillna(method='ffill').fillna(method='bfill')
                print("Missing values after filling:", df.isna().sum().sum())
            else:
                print("No missing values found.")

        except FileNotFoundError:
            print(f"Error: Data file not found at {data_path}")
            exit() # Exit if data file is not found
        except Exception as e:
            print(f"Error loading or processing data: {e}")
            exit()

        return df

    def create_time_features(self, df):
        """
        Create time-based and cyclical features from the datetime index.
        Includes day of year cyclical features and isNight based on radiation.
        Note: 'hour_sin' and 'hour_cos' are expected to be in the raw data
        as per the user's requested feature list.

        Args:
            df: DataFrame with a DatetimeIndex

        Returns:
            DataFrame with added time features
        """
        print("Creating time-based and cyclical features...")

        # Cyclical features for the day of the year (seasonal variation)
        df['day_sin'] = np.sin(2 * math.pi * df.index.dayofyear / 365.0)
        df['day_cos'] = np.cos(2 * math.pi * df.index.dayofyear / 365.0)

        # isNight calculation based on Global Radiation
        radiation_col_name = self.radiation_column # Use class attribute
        if radiation_col_name in df.columns:
            # Using a small threshold to define night
            RADIATION_THRESHOLD = 1.0 # W/m² - Adjust this threshold as needed
            df['isNight'] = (df[radiation_col_name] < RADIATION_THRESHOLD).astype(int)
            print("'isNight' feature created based on Global Radiation.")
        else:
            print(f"Warning: '{radiation_col_name}' not found. Cannot create 'isNight' feature based on irradiation.")
            # Add a placeholder column if the feature is expected later in prepare_features
            # If 'isNight' is in the user's requested features, this placeholder will be used
            df['isNight'] = 0 # Default to 0 (not night) if radiation data is missing


        return df


    def prepare_features(self, df):
        """
        Define and group features for the model based on the user's specified list
        and standard time-based engineered features (day_sin, day_cos).

        Args:
            df: DataFrame with all raw and engineered features

        Returns:
            Dictionary with feature information
        """
        # Define features to use based on the user's exact list
        # Adding day_sin and day_cos from feature engineering as they are standard.
        features = [
            self.radiation_column, # Use class attribute
            #'ClearSkyDHI',
            'ClearSkyGHI',
            'ClearSkyDNI',
            'SolarZenith [degrees]',
            'AOI [degrees]',
            #'isNight', # Included as per user's list
            'ClearSkyIndex',
            'hour_cos', # Included as per user's list
            'Temperature [degree_Celsius]',
            'WindSpeed [m s-1]',
            # Engineered day features (derived from index, not augmentation)
            'day_sin',
            'day_cos',
        ]

        # Verify all intended features exist in the dataframe after loading and engineering
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            print(f"Error: The following required features are missing from the dataset after loading and engineering: {missing_features}")
            # Since the user requested *only* these features, we should exit if critical ones are missing.
            print("Please check your data file and ensure it contains these columns.")
            exit()

        # Group features by scaling method
        # MinMaxScaler: Features that are non-negative and often bounded (like radiation, clear sky values, index)
        # StandardScaler: Features with varying ranges, potentially negative (like temperature, angles)
        # RobustScaler: Features potentially sensitive to outliers (like wind speed)
        # No scaling: Binary or already scaled cyclical features
        minmax_features = [
            self.radiation_column, # Use class attribute
            'ClearSkyDHI',
            'ClearSkyGHI',
            'ClearSkyDNI',
            'ClearSkyIndex'
        ]
        standard_features = [
            'Temperature [degree_Celsius]',
            'SolarZenith [degrees]',
            'AOI [degrees]'
        ]
        robust_features = ['WindSpeed [m s-1]']
        no_scaling_features = [
            'isNight', # Included as per user's list
            'hour_cos', # Included as per user's list
            'day_sin',
            'day_cos',
        ]

        # Ensure all features are accounted for in the scaling groups
        all_grouped_features = minmax_features + standard_features + robust_features + no_scaling_features
        if set(features) != set(all_grouped_features):
            print("Warning: Mismatch between defined 'features' list and scaling groups.")
            # Use the defined 'features' list as the definitive list for columns to select
            # Filter scaling groups to only include features that are actually in the 'features' list
            minmax_features = [f for f in minmax_features if f in features]
            standard_features = [f for f in standard_features if f in features]
            robust_features = [f for f in robust_features if f in features]
            no_scaling_features = [f for f in no_scaling_features if f in features]
            print("Adjusted scaling groups to match the defined features list.")


        # Reconstruct the 'all_features' list in a defined order for consistent scaling
        # This order is used later when creating the scaled DataFrame
        all_features_ordered = minmax_features + standard_features + robust_features + no_scaling_features
        # Ensure no duplicates and all features are included
        all_features_ordered = list(dict.fromkeys(all_features_ordered)) # Removes duplicates and preserves order

        # Final check that all original 'features' are in the ordered list (should be true after filtering)
        if set(features) != set(all_features_ordered):
            print("Error: Feature list mismatch after ordering. This should not happen. Exiting.")
            exit()
        else:
            # Use the ordered list as the definitive list of features to select from the DataFrame
            features = all_features_ordered


        target = 'power_w' # Ensure this target column exists in the data

        if target not in df.columns:
            print(f"Error: Target variable '{target}' not found in the dataset. Exiting.")
            exit()


        return {
            'all_features': features,
            'minmax_features': minmax_features,
            'standard_features': standard_features,
            'robust_features': robust_features,
            'no_scaling_features': no_scaling_features,
            'target': target
        }

    def split_data_for_tscv_and_final_test(self, df: pd.DataFrame, final_test_size_ratio: float):
        """
        Splits data into a training/validation set for TSCV and a separate final test set.

        Args:
            df: The full DataFrame with datetime index.
            final_test_size_ratio: The ratio of the data to use for the final test set (e.g., 0.15).

        Returns:
            A tuple containing:
            - train_val_df: DataFrame for training and validation (unscaled).
            - final_test_df: DataFrame for the final test set (unscaled).
        """
        df = df.sort_index() # Ensure data is sorted by time
        total_size = len(df)
        final_test_size = int(total_size * final_test_size_ratio)
        train_val_size = total_size - final_test_size

        train_val_df = df.iloc[:train_val_size].copy()
        final_test_df = df.iloc[train_val_size:].copy()

        print(f"\nTotal data size: {total_size}")
        print(f"Train/Validation set size (for TSCV): {len(train_val_df)}")
        print(f"Final Test set size: {len(final_test_df)}")

        return train_val_df, final_test_df

    def fit_and_save_scalers(self, df: pd.DataFrame, feature_info: dict):
        """
        Fits scalers on the provided DataFrame and saves them.

        Args:
            df: The DataFrame to fit scalers on (should be the training/validation data).
            feature_info: Dictionary with feature information.

        Returns:
            Dictionary of fitted scalers.
        """
        print("\nFitting scalers on training/validation data...")
        scalers = {}

        # Initialize and fit scalers for different feature groups
        if feature_info['minmax_features']:
            minmax_scaler = MinMaxScaler()
            minmax_scaler.fit(df[feature_info['minmax_features']])
            scalers['minmax'] = minmax_scaler
            joblib.dump(minmax_scaler, os.path.join(self.model_dir, f'minmax_scaler_{self.timestamp}.pkl'))
            print(f"MinMaxScaler fitted and saved for {len(feature_info['minmax_features'])} features.")

        if feature_info['standard_features']:
            standard_scaler = StandardScaler()
            standard_scaler.fit(df[feature_info['standard_features']])
            scalers['standard'] = standard_scaler
            joblib.dump(standard_scaler, os.path.join(self.model_dir, f'standard_scaler_{self.timestamp}.pkl'))
            print(f"StandardScaler fitted and saved for {len(feature_info['standard_features'])} features.")

        if feature_info['robust_features']:
            robust_scaler = RobustScaler()
            robust_scaler.fit(df[feature_info['robust_features']])
            scalers['robust'] = robust_scaler
            joblib.dump(robust_scaler, os.path.join(self.model_dir, f'robust_scaler_{self.timestamp}.pkl'))
            print(f"RobustScaler fitted and saved for {len(feature_info['robust_features'])} features.")

        # Fit target scaler
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler.fit(df[[feature_info['target']]])
        scalers['target'] = target_scaler
        joblib.dump(target_scaler, os.path.join(self.model_dir, f'target_scaler_{self.timestamp}.pkl'))
        print("Target scaler fitted and saved.")

        return scalers

    def scale_data_subset(self, df_subset: pd.DataFrame, feature_info: dict, scalers: dict) -> pd.DataFrame:
        """
        Scales a DataFrame subset using the provided fitted scalers, including the target.

        Args:
            df_subset: The DataFrame subset to scale.
            feature_info: Dictionary with feature information.
            scalers: Dictionary of fitted scalers.

        Returns:
            The scaled DataFrame subset including scaled features and scaled target.
        """
        # Create a copy to avoid modifying the original DataFrame slice
        df_subset_scaled = df_subset.copy()

        # Scale features
        if 'minmax' in scalers and feature_info['minmax_features']:
            df_subset_scaled[feature_info['minmax_features']] = scalers['minmax'].transform(df_subset_scaled[feature_info['minmax_features']])

        if 'standard' in scalers and feature_info['standard_features']:
            df_subset_scaled[feature_info['standard_features']] = scalers['standard'].transform(df_subset_scaled[feature_info['standard_features']])

        if 'robust' in scalers and feature_info['robust_features']:
            df_subset_scaled[feature_info['robust_features']] = scalers['robust'].transform(df_subset_scaled[feature_info['robust_features']])

        # Scale target
        if 'target' in scalers and feature_info['target'] in df_subset_scaled.columns:
             df_subset_scaled[[feature_info['target']]] = scalers['target'].transform(df_subset_scaled[[feature_info['target']]])
        else:
             # This case should ideally not happen if prepare_features is correct,
             # but added for robustness.
             print(f"Warning: Target column '{feature_info['target']}' not found in subset for scaling.")


        # Features in no_scaling_features are already in df_subset_scaled and not transformed

        return df_subset_scaled


    def create_sequences(self, X, y, time_steps):
        """
        Create sequences for time series forecasting.

        Args:
            X: Features DataFrame or numpy array
            y: Target array or pandas Series
            time_steps: Number of time steps to look back (sequence length)

        Returns:
            X_seq: Sequences of features (numpy array)
            y_seq: Corresponding target values (numpy array)
        """
        X_seq, y_seq = [], []

        # Convert pandas objects to numpy arrays if they aren't already
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X_vals = X.values
        else:
            X_vals = X # Assume it's already a numpy array

        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            y_vals = y.values
        else:
            y_vals = y # Assume it's already a numpy array

        # Ensure y_vals is 1D or has shape (n, 1)
        if y_vals.ndim > 1 and y_vals.shape[1] != 1:
            raise ValueError("y must be a 1D array or have shape (n, 1)")
        if y_vals.ndim == 1:
            y_vals = y_vals.reshape(-1, 1) # Reshape to (n, 1)


        for i in range(len(X_vals) - time_steps):
            # Select the sequence of features ending at i + time_steps - 1
            X_seq.append(X_vals[i : (i + time_steps)])
            # Select the target value at i + time_steps
            y_seq.append(y_vals[i + time_steps])

        return np.array(X_seq), np.array(y_seq)

    def build_model(self, input_shape, trial=None):
        """
        Build an LSTM model with optional hyperparameters from Optuna trial.
        Includes post-processing (clipping) within the model as the last layer.

        Args:
            input_shape: Shape of input data (time_steps, features)
            trial: Optuna trial object for hyperparameter optimization (optional)

        Returns:
            Compiled LSTM model with post-processing layer
        """
        # If trial is provided, use it to suggest hyperparameters
        if trial:
            # Suggest hyperparameters using Optuna
            # Number of layers
            num_lstm_layers = trial.suggest_int('num_lstm_layers', 1, 2)
            num_dense_layers = trial.suggest_int('num_dense_layers', 1, 2)

            # LSTM units for each layer - use consistent parameter ranges for each position
            lstm_units = []
            if num_lstm_layers >= 1:
                # First layer typically has more units
                lstm_units.append(trial.suggest_int('lstm_units_1', 6, 96))

            if num_lstm_layers >= 2:
                # Second layer
                lstm_units.append(trial.suggest_int('lstm_units_2', 4, 46))

            if num_lstm_layers >= 3:
                # Third layer typically has fewer units
                lstm_units.append(trial.suggest_int('lstm_units_3', 8, 64))

            # Dense units for each layer - use consistent parameter ranges for each position
            dense_units = []
            if num_dense_layers >= 1:
                # First dense layer
                dense_units.append(trial.suggest_int('dense_units_1', 4, 24))

            if num_dense_layers >= 2:
                # Second dense layer
                dense_units.append(trial.suggest_int('dense_units_2', 2, 8))

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
            batch_norm = trial.suggest_categorical('batch_norm', [True, False]) # Allow optimization to turn off batch norm

            # Update config with suggested hyperparameters
            self.config['lstm_units'] = lstm_units
            self.config['dense_units'] = dense_units
            self.config['dropout_rates'] = lstm_dropout_rates
            self.config['dense_dropout_rates'] = dense_dropout_rates
            self.config['learning_rate'] = learning_rate
            self.config['bidirectional'] = bidirectional
            self.config['batch_norm'] = batch_norm

        # Retrieve hyperparameters from self.config
        lstm_units = self.config['lstm_units']
        dense_units = self.config['dense_units']
        dropout_rates = self.config['dropout_rates']
        dense_dropout_rates = self.config['dense_dropout_rates']
        learning_rate = self.config['learning_rate']
        bidirectional = self.config['bidirectional']
        batch_norm = self.config['batch_norm']
        num_lstm_layers = len(lstm_units)
        num_dense_layers = len(dense_units)


        model = Sequential()

        # Add LSTM layers
        if num_lstm_layers > 0:
            return_sequences_first = (num_lstm_layers > 1)
            lstm_layer_instance = LSTM(units=lstm_units[0], return_sequences=return_sequences_first)

            if bidirectional:
                model.add(Bidirectional(lstm_layer_instance, input_shape=input_shape))
            else:
                model.add(LSTM(units=lstm_units[0], return_sequences=return_sequences_first, input_shape=input_shape))

            if batch_norm:
                model.add(BatchNormalization())

            # Ensure dropout_rates is not empty before accessing index 0
            dropout_rate_first = dropout_rates[0] if dropout_rates else 0.0
            if dropout_rate_first > 1e-6:
                model.add(Dropout(dropout_rate_first))


            for i in range(1, num_lstm_layers):
                is_last_lstm = (i == num_lstm_layers - 1)
                subsequent_lstm_layer_instance = LSTM(units=lstm_units[i], return_sequences=not is_last_lstm)
                if bidirectional:
                    model.add(Bidirectional(subsequent_lstm_layer_instance))
                else:
                    model.add(subsequent_lstm_layer_instance)

                if batch_norm:
                    model.add(BatchNormalization())

                # Use dropout rate corresponding to this layer index or the last rate
                dropout_rate_subsequent = dropout_rates[i] if i < len(dropout_rates) else (dropout_rates[-1] if dropout_rates else 0.0)
                if dropout_rate_subsequent > 1e-6:
                    model.add(Dropout(dropout_rate_subsequent))

        # Add Dense layers
        for i in range(num_dense_layers):
            model.add(Dense(dense_units[i], activation='relu'))

            if batch_norm:
                model.add(BatchNormalization())

            if i < num_dense_layers - 1:  # No dropout after the last dense layer
                # Use dense dropout rate corresponding to this layer index or the last rate
                dense_dropout_rate_current = dense_dropout_rates[i] if i < len(dense_dropout_rates) else (dense_dropout_rates[-1] if dense_dropout_rates else 0.0)
                if dense_dropout_rate_current > 1e-6:
                    model.add(Dropout(dense_dropout_rate_current))

        # Output layer
        model.add(Dense(1))

        # Add post-processing layer (clipping)
        # This layer ensures the *scaled* output is not negative.
        model.add(Lambda(clip_scaled_output, name='scaled_output_clipping'))


        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        return model

    def create_callbacks(self, trial=None, fold_idx=None):
        """
        Create callbacks for model training.

        Args:
            trial: Optuna trial object (optional)
            fold_idx: Index of the current fold (optional, for checkpoint naming)

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

        # Model checkpoint (only for final model training, not during hyperparameter search)
        if trial is None:
            # Checkpoint path depends on whether final training is single model or TSCV
            if fold_idx is not None: # TSCV for final training
                checkpoint_filepath = os.path.join(self.final_models_dir, f'best_model_fold_{fold_idx}_{self.timestamp}.keras')
            else: # Single model final training
                checkpoint_filepath = os.path.join(self.model_dir, f'best_model_{self.timestamp}.keras')

            model_checkpoint = ModelCheckpoint(
                filepath=checkpoint_filepath,
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
        Objective function for Optuna optimization using a single TSCV fold.

        Args:
            trial: Optuna trial object
            X_train_seq: Training sequences for the current fold
            y_train_seq: Training targets for the current fold
            X_val_seq: Validation sequences for the current fold
            y_val_seq: Validation targets for the current fold

        Returns:
            Validation loss (to be minimized) for the current fold.
        """
        tf.keras.backend.clear_session() # Clear backend for fresh start
        # Build model with hyperparameters from trial
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        model = self.build_model(input_shape, trial)

        # Create callbacks with pruning
        callbacks = self.create_callbacks(trial)

        # Train model with fewer epochs for hyperparameter search
        try:
            history = model.fit(
                X_train_seq, y_train_seq,
                epochs=20,  # Reduced epochs for hyperparameter search
                batch_size=self.batch_size,
                validation_data=(X_val_seq, y_val_seq),
                callbacks=callbacks,
                verbose=0  # Silent training during hyperparameter search
            )
            # Return the best validation loss during the trial for this fold
            validation_loss = min(history.history['val_loss'])
        except Exception as e:
            print(f"Trial {trial.number} failed due to error: {e}")
            raise optuna.exceptions.TrialPruned() # Prune trial on error

        return validation_loss

    def optimize_hyperparameters_with_tscv(self, train_val_df_scaled: pd.DataFrame, feature_info: dict, n_trials: int, n_splits: int):
        """
        Optimize hyperparameters using Optuna's Bayesian optimization with Time Series Cross-Validation.

        Args:
            train_val_df_scaled: The scaled DataFrame for training and validation.
            feature_info: Dictionary with feature information.
            n_trials: Number of optimization trials.
            n_splits: Number of splits for TimeSeriesSplit.

        Returns:
            Dictionary with optimized hyperparameters.
        """
        print(f"\nStarting hyperparameter optimization with {n_trials} trials using Optuna and {n_splits}-fold Time Series Cross-Validation...")

        # Create a study object with pruning
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
            interval_steps=1
        )
        # Use a database to store study results
        db_path = os.path.join(self.optuna_results_dir, f'optuna_study_{self.timestamp}.db')
        study = optuna.create_study(
            direction='minimize',
            pruner=pruner,
            study_name=f'lstm_lowres_study_{self.timestamp}',
            storage=f'sqlite:///{db_path}', # Store study in a database
            load_if_exists=True # Load existing study if it exists
        )
        print(f"Optuna study stored at: {db_path}")
        if len(study.trials) > 0:
            print(f"Loaded existing study with {len(study.trials)} trials.")

        # Initialize TimeSeriesSplit
        # test_size=None uses the remaining data for the test set in each split
        # gap=0 means no gap between training and testing sets in each fold
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=0)

        # Define the objective function wrapper for TSCV
        def tscv_objective(trial):
            fold_losses = []
            # Split the scaled train_val_df for TSCV
            for fold_idx, (train_index, val_index) in enumerate(tscv.split(train_val_df_scaled)):
                print(f"    Running Optuna trial {trial.number}, Fold {fold_idx + 1}/{n_splits}")

                # Get train and validation data for the current fold
                X_train_fold = train_val_df_scaled.iloc[train_index][feature_info['all_features']]
                y_train_fold = train_val_df_scaled.iloc[train_index][[feature_info['target']]]
                X_val_fold = train_val_df_scaled.iloc[val_index][feature_info['all_features']]
                y_val_fold = train_val_df_scaled.iloc[val_index][[feature_info['target']]]

                # Create sequences for the current fold
                X_train_seq_fold, y_train_seq_fold = self.create_sequences(
                    X_train_fold, y_train_fold, self.sequence_length
                )
                X_val_seq_fold, y_val_seq_fold = self.create_sequences(
                    X_val_fold, y_val_fold, self.sequence_length
                )

                # Ensure sequences are not empty
                if X_train_seq_fold.shape[0] == 0 or X_val_seq_fold.shape[0] == 0:
                    print(f"    Warning: Not enough data in fold {fold_idx + 1} to create sequences. Skipping fold.")
                    continue # Skip this fold if sequence creation fails

                # Run the objective function for this fold
                try:
                    fold_loss = self.objective(
                        trial, X_train_seq_fold, y_train_seq_fold, X_val_seq_fold, y_val_seq_fold
                    )
                    fold_losses.append(fold_loss)
                except optuna.exceptions.TrialPruned:
                    # If a trial is pruned within a fold, propagate the pruning
                    raise
                except Exception as e:
                    print(f"    Error during training fold {fold_idx + 1}: {e}")
                    # Decide how to handle errors in folds - for now, treat as a failed trial
                    raise optuna.exceptions.TrialPruned()


            if not fold_losses:
                print(f"    Warning: No successful folds for trial {trial.number}. Pruning trial.")
                raise optuna.exceptions.TrialPruned()


            # Return the average validation loss across successful folds
            avg_validation_loss = np.mean(fold_losses)
            print(f"    Optuna trial {trial.number} average validation loss: {avg_validation_loss:.6f}")
            return avg_validation_loss


        # Run the optimization, only running remaining trials if study was loaded
        remaining_trials = n_trials - len([t for t in study.trials if t.state in {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED, optuna.trial.TrialState.FAIL}])
        if remaining_trials > 0:
            print(f"Running {remaining_trials} new optimization trials...")
            study.optimize(tscv_objective, n_trials=remaining_trials, show_progress_bar=True)
        else:
            print("Optimization already completed for the specified number of trials.")


        if study.best_trial is None:
            print("No trials completed successfully.")
            return self.config # Return default config if no successful trials
        else:
            # Get best parameters
            best_params = study.best_params
            print("\nBest hyperparameters found:")
            for param, value in best_params.items():
                print(f"{param}: {value}")

            # Update config with best parameters
            num_lstm_layers = best_params.get('num_lstm_layers', len(self.config['lstm_units']))
            num_dense_layers = best_params.get('num_dense_layers', len(self.config['dense_units']))

            # Extract LSTM units, handling cases where parameter might not be in best_params
            lstm_units = [best_params.get(f'lstm_units_{i+1}', self.config['lstm_units'][i] if i < len(self.config['lstm_units']) else None) for i in range(num_lstm_layers)]
            lstm_units = [u for u in lstm_units if u is not None] # Remove None if any layer units were not found

            # Extract dense units, handling cases where parameter might not be in best_params
            dense_units = [best_params.get(f'dense_units_{i+1}', self.config['dense_units'][i] if i < len(self.config['dense_units']) else None) for i in range(num_dense_layers)]
            dense_units = [u for u in dense_units if u is not None] # Remove None if any layer units were not found

            # Extract adaptive dropout rates for LSTM and dense layers
            lstm_dropout_rates = [best_params.get(f'lstm_dropout_rate_{i+1}', self.config['dropout_rates'][i] if i < len(self.config['dropout_rates']) else None) for i in range(num_lstm_layers)]
            lstm_dropout_rates = [d for d in lstm_dropout_rates if d is not None]

            dense_dropout_rates = [best_params.get(f'dense_dropout_rate_{i+1}', self.config['dense_dropout_rates'][i] if i < len(self.config['dense_dropout_rates']) else None) for i in range(num_dense_layers - 1)] # No dropout after last dense layer
            dense_dropout_rates = [d for d in dense_dropout_rates if d is not None]

            self.config['lstm_units'] = lstm_units if lstm_units else self.config['lstm_units']
            self.config['dense_units'] = dense_units if dense_units else self.config['dense_units']
            self.config['dropout_rates'] = lstm_dropout_rates if lstm_dropout_rates else self.config['dropout_rates']
            self.config['dense_dropout_rates'] = dense_dropout_rates if dense_dropout_rates else self.config['dense_dropout_rates']
            self.config['learning_rate'] = best_params.get('learning_rate', self.config['learning_rate'])
            self.config['bidirectional'] = best_params.get('bidirectional', self.config['bidirectional'])
            self.config['batch_norm'] = best_params.get('batch_norm', self.config['batch_norm'])


            try:
                # Optuna plotting requires specific dependencies (e.g., matplotlib, plotly)
                # Ensure these are installed if you want the plots
                from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate, plot_slice
                print("Generating Optuna plots...")
                fig_history = plot_optimization_history(study)
                fig_history.write_image(os.path.join(self.optuna_results_dir, f'optimization_history_{self.timestamp}.png'))
                fig_importance = plot_param_importances(study)
                fig_importance.write_image(os.path.join(self.optuna_results_dir, f'param_importances_{self.timestamp}.png'))

                # Get all parameter names from the study
                study_params = list(study.best_params.keys())

                # Plot parallel coordinate plot
                plt.figure(figsize=(12, 8))
                plot_parallel_coordinate(study)
                plt.tight_layout()
                plt.savefig(os.path.join(self.optuna_results_dir, f'parallel_coordinate_{self.timestamp}.png'))
                plt.close() # Close figure after saving

                # Plot slice plots for key hyperparameters that exist in the study
                key_params_to_plot = ['num_lstm_layers', 'num_dense_layers', 'learning_rate'] + [p for p in study_params if 'units' in p or 'dropout_rate' in p]
                key_params_to_plot = list(dict.fromkeys(key_params_to_plot)) # Remove duplicates

                for param in key_params_to_plot:
                    if param in study_params:
                        try:
                            plt.figure(figsize=(10, 6))
                            plot_slice(study, params=[param])
                            plt.tight_layout()
                            plt.savefig(os.path.join(self.optuna_results_dir, f'slice_{param}_{self.timestamp}.png'))
                            plt.close() # Close figure after saving
                        except Exception as e_plot:
                             print(f"Warning: Could not generate slice plot for {param}. Error: {e_plot}")


                print(f"Optuna visualization plots saved to {self.optuna_results_dir}/")
            except ImportError:
                print("Warning: Optuna visualization dependencies (matplotlib, plotly) not found. Skipping plot generation.")
            except Exception as e:
                print(f"Warning: Could not generate Optuna plots. Error: {e}")


            # Save optimized hyperparameters to a text file
            best_params_path = os.path.join(self.model_dir, f'best_params_{self.timestamp}.txt')
            with open(best_params_path, 'w', encoding='utf-8') as f:
                f.write("Optimized Hyperparameters:\n")
                for param, value in self.config.items():
                    f.write(f"{param}: {value}\n")
            print(f"Best hyperparameters saved to {best_params_path}")

            # Save study for later analysis
            study_save_path = os.path.join(self.optuna_results_dir, f'study_{self.timestamp}.pkl')
            joblib.dump(study, study_save_path)
            print(f"Optuna study object saved to {study_save_path}")


            return self.config # Return the updated config with best params

    def apply_zero_radiation_postprocessing(self, y_pred_scaled_raw: np.ndarray, original_df_slice: pd.DataFrame, radiation_column: str, target_scaler: MinMaxScaler) -> np.ndarray:
        """
        Applies post-processing: sets predicted power to 0 if future Global Radiation is 0.

        Args:
            y_pred_scaled_raw: Raw predictions from the model (scaled), shape (n_predictions, 1).
            original_df_slice: The original (unscaled) DataFrame slice corresponding to the predictions.
                               Must be aligned with y_pred_scaled_raw.
            radiation_column: The name of the Global Radiation column in original_df_slice.
            target_scaler: The scaler used for the target variable, needed to transform 0 back to scaled space.

        Returns:
            Post-processed predictions (scaled), shape (n_predictions, 1).
        """
        print("\nApplying zero radiation post-processing...")

        if radiation_column not in original_df_slice.columns:
            print(f"Warning: Radiation column '{radiation_column}' not found in original data for post-processing.")
            return y_pred_scaled_raw.copy() # Return a copy of raw predictions if radiation data is missing

        # Ensure lengths match - y_pred_scaled_raw should have the same number of predictions
        # as rows in original_df_slice if sliced correctly.
        if len(y_pred_scaled_raw) != len(original_df_slice):
             print(f"Warning: Length mismatch between predictions ({len(y_pred_scaled_raw)}) and original data slice ({len(original_df_slice)}). Skipping zero radiation post-processing.")
             return y_pred_scaled_raw.copy() # Return a copy of raw predictions

        # Create a boolean mask where Global Radiation is zero (or very close to zero)
        # at the prediction timestamps in the original data slice.
        # Use a small threshold to account for floating point issues if necessary.
        RADIATION_THRESHOLD = 1.0 # W/m² - Adjust this threshold as needed
        zero_radiation_mask = (original_df_slice[radiation_column] < RADIATION_THRESHOLD)

        # Get the scaled value for zero power
        # This assumes the target scaler maps 0W to a specific scaled value (often 0)
        scaled_zero_value = target_scaler.transform([[0.0]])[0][0] # Transform 0W to scaled value

        # Apply the post-processing: set prediction to scaled_zero_value where radiation is zero
        y_pred_scaled_postprocessed = y_pred_scaled_raw.copy() # Work on a copy
        # Apply the mask. The mask needs to be reshaped to match the prediction shape (n_predictions, 1)
        y_pred_scaled_postprocessed[zero_radiation_mask.values.reshape(-1, 1)] = scaled_zero_value

        print(f"Applied zero radiation post-processing to {zero_radiation_mask.sum()} predictions.")

        return y_pred_scaled_postprocessed


    def evaluate_model(self, y_test_scaled, y_pred_scaled_raw, y_pred_scaled_postprocessed, target_scaler):
        """
        Evaluate the model and return metrics using raw and post-processed predictions.
        This function now takes scaled predictions directly.

        Args:
            y_test_scaled: True target values (scaled).
            y_pred_scaled_raw: Raw predictions from the model (scaled).
            y_pred_scaled_postprocessed: Post-processed predictions (scaled).
            target_scaler: Scaler for the target variable (needed for inverse transform).


        Returns:
            Dictionary of evaluation metrics and actual/predicted values.
        """
        print("\nCalculating evaluation metrics...")

        # Ensure shapes are consistent
        y_test_scaled = y_test_scaled.reshape(-1, 1)
        y_pred_scaled_raw = y_pred_scaled_raw.reshape(-1, 1)
        y_pred_scaled_postprocessed = y_pred_scaled_postprocessed.reshape(-1, 1)


        # Calculate metrics on SCALED data using POST-PROCESSED predictions
        mse_scaled = mean_squared_error(y_test_scaled, y_pred_scaled_postprocessed)
        rmse_scaled = np.sqrt(mse_scaled)
        mae_scaled = mean_absolute_error(y_test_scaled, y_pred_scaled_postprocessed)
        r2_scaled = r2_score(y_test_scaled, y_pred_scaled_postprocessed)

        # Recalculate MAPE and SMAPE using postprocessed scaled predictions
        epsilon = 1e-8
        mape_scaled = np.mean(np.abs((y_test_scaled - y_pred_scaled_postprocessed) / (y_test_scaled + epsilon))) * 100
        smape_scaled = np.mean(2.0 * np.abs(y_pred_scaled_postprocessed - y_test_scaled) / (np.abs(y_pred_scaled_postprocessed) + np.abs(y_test_scaled) + epsilon)) * 100


        print("\nModel Evaluation (Scaled, Post-processed):")
        print(f"Mean Squared Error (MSE): {mse_scaled:.6f}")
        print(f"Root Mean Squared Error (RMSE): {rmse_scaled:.6f}")
        print(f"Mean Absolute Error (MAE): {mae_scaled:.6f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape_scaled:.2f}%")
        print(f"Symmetric Mean Absolute Percentage Error (SMAPE): {smape_scaled:.2f}%")
        print(f"R² Score: {r2_scaled:.4f}")

        results = {
            'mse_scaled_postprocessed': mse_scaled,
            'rmse_scaled_postprocessed': rmse_scaled,
            'mae_scaled_postprocessed': mae_scaled,
            'smape_scaled_postprocessed': smape_scaled,
            'r2_scaled_postprocessed': r2_scaled,
            'y_test_scaled': y_test_scaled,
            'y_pred_scaled_raw': y_pred_scaled_raw, # Keep raw predictions for comparison if needed
            'y_pred_scaled_postprocessed': y_pred_scaled_postprocessed
        }

        # Inverse transform original test values and POST-PROCESSED predictions
        y_test_inv = target_scaler.inverse_transform(y_test_scaled)
        y_pred_inv_postprocessed = target_scaler.inverse_transform(y_pred_scaled_postprocessed)


        # Calculate metrics on ORIGINAL scale using POST-PROCESSED predictions
        mse_postprocessed = mean_squared_error(y_test_inv, y_pred_inv_postprocessed)
        rmse_postprocessed = np.sqrt(mse_postprocessed)
        mae_postprocessed = mean_absolute_error(y_test_inv, y_pred_inv_postprocessed)
        r2_postprocessed = r2_score(y_test_inv, y_pred_inv_postprocessed)

        # Calculate MAPE and SMAPE on original scale using POST-PROCESSED predictions
        epsilon = 1e-8
        mape_postprocessed = np.mean(np.abs((y_test_inv - y_pred_inv_postprocessed) / (y_test_inv + epsilon))) * 100
        smape_postprocessed = np.mean(2.0 * np.abs(y_pred_inv_postprocessed - y_test_inv) / (np.abs(y_pred_inv_postprocessed) + np.abs(y_test_inv) + epsilon)) * 100


        print("\nModel Evaluation (Original Scale, Post-processed):")
        print(f"Mean Squared Error (MSE): {mse_postprocessed:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse_postprocessed:.2f}")
        print(f"Mean Absolute Error (MAE): {mae_postprocessed:.2f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape_postprocessed:.2f}%")
        print(f"Symmetric Mean Absolute Percentage Error (SMAPE): {smape_postprocessed:.2f}%")
        print(f"R² Score: {r2_postprocessed:.4f}")

        results.update({
            'mse_postprocessed': mse_postprocessed,
            'rmse_postprocessed': rmse_postprocessed,
            'mae_postprocessed': mae_postprocessed,
            'mape_postprocessed': mape_postprocessed,
            'smape_postprocessed': smape_postprocessed,
            'r2_postprocessed': r2_postprocessed,
            'y_test_inv': y_test_inv,
            'y_pred_inv_postprocessed': y_pred_inv_postprocessed # Store post-processed inverse predictions
        })

        # Optionally, calculate and store raw metrics (before post-processing) for comparison
        # This requires inverse transforming the raw scaled predictions
        try:
            y_pred_inv_raw = target_scaler.inverse_transform(y_pred_scaled_raw)
            mse_raw = mean_squared_error(y_test_inv, y_pred_inv_raw)
            rmse_raw = np.sqrt(mse_raw)
            mae_raw = mean_absolute_error(y_test_inv, y_pred_inv_raw)
            r2_raw = r2_score(y_test_inv, y_pred_inv_raw)

            epsilon = 1e-8
            mape_raw = np.mean(np.abs((y_test_inv - y_pred_inv_raw) / (y_test_inv + epsilon))) * 100
            smape_raw = np.mean(2.0 * np.abs(y_pred_inv_raw - y_test_inv) / (np.abs(y_pred_inv_raw) + np.abs(y_test_inv) + epsilon)) * 100

            print("\nModel Evaluation (Original Scale, Raw):")
            print(f"Mean Squared Error (MSE): {mse_raw:.2f}")
            print(f"Root Mean Squared Error (RMSE): {rmse_raw:.2f}")
            print(f"Mean Absolute Error (MAE): {mae_raw:.2f}")
            print(f"Mean Absolute Percentage Error (MAPE): {mape_raw:.2f}%")
            print(f"Symmetric Mean Absolute Percentage Error (SMAPE): {smape_raw:.2f}%")
            print(f"R² Score: {r2_raw:.4f}")

            results.update({
                'mse_raw': mse_raw,
                'rmse_raw': rmse_raw,
                'mae_raw': mae_raw,
                'mape_raw': mape_raw,
                'smape_raw': smape_raw,
                'r2_raw': r2_raw,
                'y_pred_inv_raw': y_pred_inv_raw # Store raw inverse predictions
            })
        except Exception as e:
             print(f"Warning: Could not calculate raw metrics: {e}")
             # Handle cases where inverse transform might fail or data is empty


        return results

    def save_model_summary(self, model_summary_str, feature_info, evaluation_results, data_path):
        """
        Saves a summary of the model architecture, configuration, and evaluation results.
        Accepts model summary as a string now since multiple models are trained.
        """
        summary_path = os.path.join(self.model_dir, f'model_summary_{self.timestamp}.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Model Summary - Timestamp: {self.timestamp}\n")
            f.write("-" * 30 + "\n")
            f.write("Input Data:\n")
            f.write(f"    Data Path: {data_path}\n")
            f.write(f"    Sequence Length: {self.sequence_length}\n")
            f.write(f"    Features Used ({len(feature_info['all_features'])}): {feature_info['all_features']}\n")
            f.write("-" * 30 + "\n")
            f.write("Model Configuration (from best Optuna trial or default):\n")
            for key, value in self.config.items():
                f.write(f"    {key}: {value}\n")
            f.write("-" * 30 + "\n")
            f.write("Model Architecture (Summary from one of the final trained models):\n")
            f.write(model_summary_str + '\n') # Write the provided summary string
            f.write("-" * 30 + "\n")
            f.write("Evaluation Results (Final Test Set):\n") # Updated title
            # Print post-processed metrics if available, otherwise raw metrics
            if 'rmse_postprocessed' in evaluation_results:
                f.write("    Metrics (Post-processed, Original Scale):\n")
                f.write(f"    RMSE: {evaluation_results['rmse_postprocessed']:.2f}\n")
                f.write(f"    MAE: {evaluation_results['mae_postprocessed']:.2f}\n")
                f.write(f"    MAPE: {evaluation_results['mape_postprocessed']:.2f}%\n")
                f.write(f"    SMAPE: {evaluation_results['smape_postprocessed']:.2f}%\n")
                f.write(f"    R²: {evaluation_results['r2_postprocessed']:.4f}\n")
            elif 'rmse_raw' in evaluation_results: # Print raw metrics if calculated
                 f.write("    Metrics (Raw, Original Scale):\n")
                 f.write(f"    RMSE: {evaluation_results['rmse_raw']:.2f}\n")
                 f.write(f"    MAE: {evaluation_results['mae_raw']:.2f}\n")
                 f.write(f"    MAPE: {evaluation_results['mape_raw']:.2f}%")
                 f.write(f"    SMAPE: {evaluation_results['smape_raw']:.2f}%")
                 f.write(f"    R²: {evaluation_results['r2_raw']:.4f}\n")
            else: # Fallback to scaled raw metrics if nothing else
                 f.write("    Metrics (Scaled Raw Fallback):\n")
                 f.write(f"    RMSE: {evaluation_results['rmse_scaled']:.6f}\n")
                 f.write(f"    MAE: {evaluation_results['mae_scaled']:.6f}\n")
                 f.write(f"    MAPE: {evaluation_results['mape_scaled']:.2f}%")
                 f.write(f"    SMAPE: {evaluation_results['smape_scaled']:.2f}%")
                 f.write(f"    R²: {evaluation_results['r2_scaled']:.4f}\n")


            f.write("-" * 30 + "\n")
            f.write(f"Scalers saved to: {self.model_dir}/*_scaler_{self.timestamp}.pkl\n")
            f.write(f"Final TSCV models saved to: {self.final_models_dir}/\n") # Updated model save location
            f.write(f"Results plots saved to: {self.results_dir}/lstm_lowres_*{self.timestamp}.png\n")
            if os.path.exists(os.path.join(self.optuna_results_dir, f'optuna_study_{self.timestamp}.db')):
                 f.write(f"Optuna study database: {self.optuna_results_dir}/optuna_study_{self.timestamp}.db\n")
                 # Also list plot files if they were generated
                 if os.path.exists(os.path.join(self.optuna_results_dir, f'optimization_history_{self.timestamp}.png')):
                     f.write(f"Optuna plots saved to: {self.optuna_results_dir}/*_{self.timestamp}.png\n")


        print(f"\nModel summary saved to {summary_path}")


    def plot_results(self, history, evaluation_results):
        """
        Plot and save training history and prediction results.
        Plots post-processed predictions if available.

        Args:
            history: Training history object (from the last fold trained)
            evaluation_results: Dictionary from model evaluation (using averaged predictions)
        """
        print(f"Saving training history and prediction plots to {self.results_dir}/")

        # Plot training history (from the last trained fold)
        if history: # Check if history object is available
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss (Last TSCV Fold)') # Updated title
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(history.history['mae'], label='Training MAE')
            plt.plot(history.history['val_mae'], label='Validation MAE')
            plt.title('Model MAE (Last TSCV Fold)') # Updated title
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f'lstm_lowres_history_{self.timestamp}.png'))
            # plt.close() # Removed plt.close()
        else:
            print("Warning: Training history not available for plotting.")


        # Determine which predictions to plot on scaled plots (post-processed is preferred)
        y_pred_scaled_to_plot = evaluation_results.get('y_pred_scaled_postprocessed', evaluation_results.get('y_pred_scaled_raw')) # Use get with default None or raw
        if y_pred_scaled_to_plot is None or 'y_test_scaled' not in evaluation_results:
             print("Error: No scaled predictions or actuals available for plotting.")
             return # Exit plot function if no predictions


        plot_title_suffix_scaled = ' (Scaled, Post-processed, Averaged)' if 'y_pred_scaled_postprocessed' in evaluation_results else ' (Scaled, Raw, Averaged)'


        # Plot predictions vs actual (scaled) - Sample a smaller portion for clarity if needed
        plt.figure(figsize=(14, 7))

        sample_size = min(20000, len(evaluation_results['y_test_scaled']))
        indices = np.arange(sample_size)

        plt.plot(indices, evaluation_results['y_test_scaled'][:sample_size], 'b-', label='Actual (Scaled)')
        # Update label based on whether post-processing was applied
        pred_label_scaled = 'Predicted (Scaled, Post-processed, Averaged)' if 'y_pred_scaled_postprocessed' in evaluation_results else 'Predicted (Scaled, Raw, Averaged)'
        plt.plot(indices, y_pred_scaled_to_plot[:sample_size], 'r-', label=pred_label_scaled)
        plt.title(f'Actual vs Predicted PV Power{plot_title_suffix_scaled} - Sample ({sample_size} points)')
        plt.xlabel(f'Time Steps ({self.sequence_length}-step lookback)')
        plt.ylabel('Power Output (Scaled)')
        plt.legend()
        plt.grid(True)

        plt.savefig(os.path.join(self.results_dir, f'lstm_lowres_predictions_scaled_{self.timestamp}.png'))
        # plt.close() # Removed plt.close()


        # Determine which inverse-transformed predictions to plot (post-processed is preferred)
        y_pred_inv_to_plot = evaluation_results.get('y_pred_inv_postprocessed', evaluation_results.get('y_pred_inv_raw')) # Use get with default None or raw inverse
        if y_pred_inv_to_plot is None or 'y_test_inv' not in evaluation_results:
             print("Warning: No inverse-transformed predictions or actuals available for plotting on original scale.")
             # Still try to plot scaled scatter if no inverse data at all
             if 'y_test_scaled' in evaluation_results and 'y_pred_scaled_to_plot' in locals(): # Check if scaled data is available
                 plt.figure(figsize=(10, 8))
                 scatter_sample_size = min(5000, len(evaluation_results['y_test_scaled']))
                 scatter_indices = np.random.choice(len(evaluation_results['y_test_scaled']), scatter_sample_size, replace=False)

                 plt.scatter(evaluation_results['y_test_scaled'][scatter_indices], y_pred_scaled_to_plot[scatter_indices], alpha=0.5, s=5)
                 max_val = max(evaluation_results['y_test_scaled'].max(), y_pred_scaled_to_plot.max())
                 min_val = min(evaluation_results['y_test_scaled'].min(), y_pred_scaled_to_plot.min())
                 plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

                 plt.title(f'Actual vs Predicted (Scaled) - Scatter Plot ({scatter_sample_size} points sampled)')
                 plt.xlabel('Actual Power Output (Scaled)')
                 plt.ylabel('Predicted Power Output (Scaled)')
                 plt.grid(True)
                 plt.axis('equal')
                 plt.gca().set_aspect('equal', adjustable='box')

                 plt.savefig(os.path.join(self.results_dir, f'lstm_lowres_scatter_scaled_{self.timestamp}.png'))
                 # plt.close() # Removed plt.close()
             return # Exit plot function if no inverse transformed data

        # Plot predictions vs actual (original scale)
        plt.figure(figsize=(14, 7))

        sample_size_inv = min(1000, len(evaluation_results['y_test_inv']))
        indices_inv = np.arange(sample_size_inv)

        plt.plot(indices_inv, evaluation_results['y_test_inv'][:sample_size_inv], 'b-', label='Actual Power Output')
        # Update label based on whether post-processing was applied
        pred_label_inv = 'Predicted Power Output (Post-processed, Averaged)' if 'y_pred_inv_postprocessed' in evaluation_results else 'Predicted Power Output (Raw, Averaged)'
        plt.plot(indices_inv, y_pred_inv_to_plot[:sample_size_inv], 'r-', label=pred_label_inv)
        plt.title(f'Actual vs Predicted PV Power (W) - Sample ({sample_size_inv} points)')
        plt.xlabel(f'Time Steps ({self.sequence_length}-step lookback)')
        plt.ylabel('Power Output (W)')
        plt.legend()
        plt.grid(True)

        plt.savefig(os.path.join(self.results_dir, f'lstm_lowres_predictions_{self.timestamp}.png'))
        # plt.close() # Removed plt.close()

        # Create scatter plot (original scale)
        plt.figure(figsize=(10, 8))
        scatter_sample_size = min(5000, len(evaluation_results['y_test_inv']))
        scatter_indices = np.random.choice(len(evaluation_results['y_test_inv']), scatter_sample_size, replace=False)

        # Update scatter label based on whether post-processing was applied
        plt.scatter(evaluation_results['y_test_inv'][scatter_indices], y_pred_inv_to_plot[scatter_indices], alpha=0.5, s=5, label=pred_label_inv)
        max_val = max(evaluation_results['y_test_inv'].max(), y_pred_inv_to_plot.max())
        min_val = min(evaluation_results['y_test_inv'].min(), y_pred_inv_to_plot.min())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

        plt.title(f'Actual vs Predicted PV Power (W) - Scatter Plot ({scatter_sample_size} points sampled)')
        plt.xlabel('Actual Power Output (W)')
        plt.ylabel('Predicted Power Output (W)')
        plt.grid(True)
        plt.axis('equal')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend() # Add legend to scatter plot
        plt.savefig(os.path.join(self.results_dir, f'lstm_lowres_scatter_{self.timestamp}.png'))
        # plt.close() # Removed plt.close()

        print("Plots saved.")

    def plot_full_test_predictions(self, evaluation_results, original_test_df):
        """
        Plots the actual vs predicted power output for the entire test dataset.

        Args:
            evaluation_results: Dictionary containing 'y_test_inv' and
                                'y_pred_inv_postprocessed' (or 'y_pred_inv_raw').
            original_test_df: The original (unscaled) DataFrame for the test set.
                              Used to get the datetime index for plotting.
                              Must be the full test set DataFrame returned by split_data_for_tscv_and_final_test.
        """
        print("\nPlotting actual vs predicted for the entire test set...")

        y_test_inv = evaluation_results.get('y_test_inv')
        y_pred_inv_to_plot = evaluation_results.get('y_pred_inv_postprocessed', evaluation_results.get('y_pred_inv_raw'))

        if y_test_inv is None or y_pred_inv_to_plot is None:
            print("Warning: Actual or predicted inverse transformed data not available for full test plot.")
            return

        # The predictions correspond to the timestamps starting after the sequence length
        # in the original test DataFrame.
        # The length of y_test_inv and y_pred_inv_to_plot should be len(original_test_df) - sequence_length.
        # We need to get the index slice from original_test_df that matches these predictions.
        # Ensure original_test_df is long enough to create sequences
        if len(original_test_df) < self.sequence_length:
             print(f"Warning: Original test data is too short ({len(original_test_df)}) to create sequences with length {self.sequence_length}. Cannot plot full test predictions.")
             return

        prediction_index = original_test_df.index[self.sequence_length:]

        # Ensure the index length matches the prediction length
        if len(prediction_index) != len(y_test_inv):
            print(f"Warning: Index length ({len(prediction_index)}) does not match prediction length ({len(y_test_inv)}). Cannot plot full test predictions.")
            return

        plt.figure(figsize=(18, 8)) # Larger figure for the full test set
        plt.plot(prediction_index, y_test_inv, label='Actual Power Output')

        # Update label based on whether post-processing was applied
        pred_label_inv = 'Predicted Power Output (Post-processed, Averaged)' if 'y_pred_inv_postprocessed' in evaluation_results else 'Predicted Power Output (Raw, Averaged)'
        plt.plot(prediction_index, y_pred_inv_to_plot, label=pred_label_inv, alpha=0.7) # Use alpha for potentially dense plots

        plt.title('Actual vs Predicted PV Power (W) - Full Test Set (Averaged Predictions)') # Updated title
        plt.xlabel('Time')
        plt.ylabel('Power Output (W)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45) # Rotate x-axis labels for better readability
        plt.tight_layout() # Adjust layout to prevent labels overlapping

        plot_path = os.path.join(self.results_dir, f'lstm_lowres_predictions_full_test_{self.timestamp}.png')
        plt.savefig(plot_path)
        # plt.close() # Removed plt.close()

        print(f"Full test prediction plot saved to {plot_path}")

    def plot_tscv_splits(self, train_val_df: pd.DataFrame, n_splits: int, sequence_length: int):
        """
        Visualizes the splits generated by TimeSeriesSplit for the training/validation data.

        Args:
            train_val_df: The DataFrame used for training and validation (unscaled).
            n_splits: The number of splits for TimeSeriesSplit.
            sequence_length: The sequence length used for creating lookback windows.
        """
        print(f"\nVisualizing {n_splits}-fold Time Series Cross-Validation splits...")

        tscv = TimeSeriesSplit(n_splits=n_splits, gap=0)
        fig, ax = plt.subplots(figsize=(15, 6))

        # Define colors for training and validation sets
        cmap_cv = plt.cm.coolwarm
        train_color = cmap_cv(0.02) # Blueish
        val_color = cmap_cv(0.8) # Reddish

        # Generate the training/testing visualizations for each CV split
        for ii, (tr, tt) in enumerate(tscv.split(X=train_val_df)):
            # Fill in indices with the training/test groups
            indices = np.array([np.nan] * len(train_val_df))
            indices[tr] = 0 # Mark training indices
            indices[tt] = 1 # Mark validation indices

            # Visualize the results
            ax.scatter(
                range(len(indices)),
                [ii + 0.5] * len(indices),
                c=indices,
                marker="_",
                lw=30, # Increased line width for clarity
                cmap=cmap_cv,
                vmin=-0.2,
                vmax=1.2,
            )

            # Optionally, mark the actual prediction points in the validation set
            # These start after the sequence_length in the validation fold
            if len(tt) > sequence_length:
                 prediction_indices_in_fold = tt[sequence_length:]
                 ax.scatter(
                     prediction_indices_in_fold,
                     [ii + 0.5] * len(prediction_indices_in_fold),
                     c='green', # Use a different color for prediction points
                     marker="_",
                     lw=30,
                     label='Validation Prediction Points' if ii == 0 else "", # Label only once
                     alpha=0.8 # Slightly transparent
                 )


        # Formatting
        yticklabels = [f'Fold {i+1}' for i in range(n_splits)]
        ax.set(
            yticks=np.arange(n_splits) + 0.5,
            yticklabels=yticklabels,
            xlabel="Sample index",
            ylabel="CV Fold",
            ylim=[n_splits + 0.2, -0.2],
            xlim=[0, len(train_val_df)],
        )
        ax.set_title(f'Time Series Cross-Validation Splits (n_splits={n_splits}, Sequence Length={sequence_length})', fontsize=15)

        # Create custom legend
        legend_elements = [
            Patch(color=train_color, label='Training set'),
            Patch(color=val_color, label='Validation set'),
            Patch(color='green', label=f'Validation Prediction Points (after {sequence_length} lookback)') # Add label for prediction points
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

        plt.tight_layout() # Adjust layout to prevent labels overlapping
        plot_path = os.path.join(self.results_dir, f'lstm_lowres_tscv_splits_{self.timestamp}.png')
        plt.savefig(plot_path)
        # plt.close() # Removed plt.close()
        print(f"TSCV splits visualization saved to {plot_path}")


    def run_pipeline(self, data_path, optimize_hyperparams=True, n_trials=50, n_splits_tscv=None, final_test_size_ratio=0.15):
        """
        Run the full LSTM forecasting pipeline for low-resolution data with conditional TSCV.

        Args:
            data_path: Path to the 1-hour resolution data file.
            optimize_hyperparams: Whether to perform hyperparameter optimization with TSCV.
            n_trials: Number of Optuna optimization trials.
            n_splits_tscv: Number of splits for TimeSeriesSplit during optimization and final training.
                           If None or <= 1, a single model is trained on the entire train/val set.
            final_test_size_ratio: The ratio of the data to use for the final test set.

        Returns:
            Dictionary of evaluation metrics on the final test set.
        """
        print(f"Running LSTM forecasting pipeline for low-resolution (1-hour) data from {data_path}")

        # --- Data Loading and Preparation ---
        # Load the raw data
        df = self.load_data(data_path)

        # Create time features (day_sin, day_cos, isNight)
        df = self.create_time_features(df)

        # Prepare features (defines which columns to use and how to scale)
        feature_info = self.prepare_features(df)

        # Select only the features and target defined in prepare_features before splitting/scaling
        required_cols = feature_info['all_features'] + [feature_info['target']]
        missing_required_cols = [col for col in required_cols if col not in df.columns]
        if missing_required_cols:
            print(f"Error: The following required columns (features or target) are missing before splitting/scaling: {missing_required_cols}. Exiting.")
            exit()

        df_processed = df[required_cols].copy()
        # Keep a copy of df_processed *before* the final split for plotting index later
        df_processed_before_split = df_processed.copy()


        # Split data into training/validation and final test sets
        train_val_df, final_test_df = self.split_data_for_tscv_and_final_test(df_processed, final_test_size_ratio)

        # Fit scalers ONLY on the training/validation data
        scalers = self.fit_and_save_scalers(train_val_df, feature_info)

        # Scale the training/validation data
        train_val_df_scaled = self.scale_data_subset(train_val_df, feature_info, scalers)

        # Clear original dataframes to free memory (except final_test_df which is needed later)
        del df, df_processed
        # --- End of Data Loading and Preparation ---

        # Determine if TSCV should be used for final training
        use_tscv_for_final_training = (n_splits_tscv is not None and n_splits_tscv > 1)

        # --- Visualize TSCV Splits (only if optimization is on and TSCV is used) ---
        if optimize_hyperparams and use_tscv_for_final_training:
             # Visualize the TSCV splits using the unscaled train_val_df for indexing clarity
             self.plot_tscv_splits(train_val_df, n_splits_tscv, self.sequence_length)
        # --- End of Visualize TSCV Splits ---


        # Perform hyperparameter optimization with TSCV if requested
        if optimize_hyperparams:
            print("\nPerforming hyperparameter optimization with TSCV...")
            # Optimize using the scaled training/validation data
            # Note: Optimization always uses TSCV with n_splits_tscv if optimize is True
            # We use the provided n_splits_tscv for optimization splits
            optimization_n_splits = n_splits_tscv if n_splits_tscv is not None and n_splits_tscv > 1 else 5 # Default splits for optimization if not specified or <= 1
            if optimization_n_splits <= 1:
                 print("Warning: n_splits_tscv for optimization is <= 1. Optimization will not use TSCV.")
                 # If optimization is requested but n_splits_tscv is <= 1, we could
                 # potentially skip optimization or run a single trial.
                 # For now, we'll proceed with a single 'fold' in the optimization objective.
                 optimization_n_splits = 2 # Ensure at least 2 splits for TSCV logic in objective

            self.optimize_hyperparameters_with_tscv(
                train_val_df_scaled, feature_info, n_trials=n_trials, n_splits=optimization_n_splits
            )
            print(f"\nOptimized hyperparameters loaded into configuration.")
        else:
            print("\nHyperparameter optimization skipped. Using default or previously loaded configuration.")


        # --- Final Model Training ---
        if use_tscv_for_final_training:
            print(f"\nTraining final models using {n_splits_tscv}-fold Time Series Cross-Validation on the training/validation set...")

            # Initialize TimeSeriesSplit for the final training phase
            tscv_final_training = TimeSeriesSplit(n_splits=n_splits_tscv, gap=0)

            # Prepare lists to store predictions from each fold's model on the final test set
            final_test_predictions_scaled_raw_folds = []
            final_test_predictions_scaled_postprocessed_folds = []
            training_histories = [] # To store history from each fold (can plot the last one)

            # Scale the final test data once before the loop
            final_test_df_scaled = self.scale_data_subset(final_test_df, feature_info, scalers)
            X_final_test_seq, y_final_test_scaled = self.create_sequences(
                 final_test_df_scaled[feature_info['all_features']],
                 final_test_df_scaled[[feature_info['target']]],
                 self.sequence_length
            )

            # Ensure final test sequences are not empty before proceeding
            if X_final_test_seq.shape[0] == 0:
                 print("Error: Not enough data in the final test set to create sequences for evaluation. Cannot perform final training and evaluation.")
                 # Clear scaled final_test_df
                 del final_test_df_scaled
                 # Clear original final_test_df
                 del final_test_df
                 # Display plots generated so far (like TSCV splits if optimization was on)
                 plt.show()
                 return {} # Return empty results

            # Slice the original final_test_df for post-processing logic, aligned with predictions
            original_final_test_df_slice_for_eval = final_test_df.iloc[self.sequence_length:].copy()


            for fold_idx, (train_index, val_index) in enumerate(tscv_final_training.split(train_val_df_scaled)):
                print(f"\nTraining model for Final TSCV Fold {fold_idx + 1}/{n_splits_tscv}...")

                # Get train and validation data for the current fold (scaled)
                X_train_fold = train_val_df_scaled.iloc[train_index][feature_info['all_features']]
                y_train_fold = train_val_df_scaled.iloc[train_index][[feature_info['target']]]
                X_val_fold = train_val_df_scaled.iloc[val_index][feature_info['all_features']]
                y_val_fold = train_val_df_scaled.iloc[val_index][[feature_info['target']]]

                # Create sequences for the current fold
                X_train_seq_fold, y_train_seq_fold = self.create_sequences(
                    X_train_fold, y_train_fold, self.sequence_length
                )
                X_val_seq_fold, y_val_seq_fold = self.create_sequences(
                    X_val_fold, y_val_fold, self.sequence_length
                )

                # Ensure sequences are not empty for this fold
                if X_train_seq_fold.shape[0] == 0 or X_val_seq_fold.shape[0] == 0:
                    print(f"Warning: Not enough data in Final TSCV Fold {fold_idx + 1} to create sequences. Skipping training for this fold.")
                    continue # Skip this fold

                # Build a fresh model for each fold with the best (or default) hyperparameters
                input_shape = (X_train_seq_fold.shape[1], X_train_seq_fold.shape[2])
                model = self.build_model(input_shape)

                # Create callbacks for training this fold
                callbacks = self.create_callbacks(fold_idx=fold_idx)

                # Train the model for this fold
                history = model.fit(
                    X_train_seq_fold, y_train_seq_fold,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    validation_data=(X_val_seq_fold, y_val_seq_fold),
                    callbacks=callbacks,
                    verbose=1
                )
                training_histories.append(history) # Store history

                # Load the best model for this fold saved by the checkpoint
                fold_best_model_path = os.path.join(self.final_models_dir, f'best_model_fold_{fold_idx}_{self.timestamp}.keras')
                if os.path.exists(fold_best_model_path):
                    print(f"Loading best model for Final TSCV Fold {fold_idx + 1} from {fold_best_model_path}")
                    model = tf.keras.models.load_model(fold_best_model_path, safe_mode=False, custom_objects={'clip_scaled_output': clip_scaled_output})
                else:
                    print(f"Warning: Best model checkpoint not found for Final TSCV Fold {fold_idx + 1}. Using the model from the last training epoch for this fold.")


                # Make predictions on the *final test set* using the model trained on this fold
                # Need to ensure X_final_test_seq is created and has data before predicting
                if X_final_test_seq.shape[0] > 0:
                    fold_predictions_scaled_raw = model.predict(X_final_test_seq)
                    fold_predictions_scaled_raw = fold_predictions_scaled_raw.reshape(-1, 1)

                    # Apply post-processing to the predictions from this fold
                    # Pass the radiation column name using the class attribute
                    fold_predictions_scaled_postprocessed = self.apply_zero_radiation_postprocessing(
                        fold_predictions_scaled_raw,
                        original_final_test_df_slice_for_eval,
                        self.radiation_column, # Pass radiation column name using class attribute
                        scalers['target'] # Pass target scaler
                    )

                    final_test_predictions_scaled_raw_folds.append(fold_predictions_scaled_raw)
                    final_test_predictions_scaled_postprocessed_folds.append(fold_predictions_scaled_postprocessed)
                else:
                     print(f"Warning: Final test set sequences are empty. Cannot make predictions for Final TSCV Fold {fold_idx + 1}.")


                # Clear model and data for this fold to free memory
                del model, X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_train_seq_fold, y_train_seq_fold, X_val_seq_fold, y_val_seq_fold
                tf.keras.backend.clear_session() # Clear backend session after each fold

            # Clear scaled train_val_df after all folds are processed
            del train_val_df_scaled

            # --- Averaging Predictions and Final Evaluation (after TSCV training) ---
            evaluation_results = {}
            last_history = training_histories[-1] if training_histories else None # Get history from the last trained fold

            if final_test_predictions_scaled_postprocessed_folds:
                print("\nAveraging predictions from all trained folds...")
                # Average the scaled predictions across all folds
                avg_final_test_predictions_scaled_raw = np.mean(final_test_predictions_scaled_raw_folds, axis=0)
                avg_final_test_predictions_scaled_postprocessed = np.mean(final_test_predictions_scaled_postprocessed_folds, axis=0)

                # Inverse transform the averaged predictions
                avg_final_test_predictions_inv_raw = scalers['target'].inverse_transform(avg_final_test_predictions_scaled_raw)
                avg_final_test_predictions_inv_postprocessed = scalers['target'].inverse_transform(avg_final_test_predictions_scaled_postprocessed)

                # Calculate final evaluation metrics using the averaged predictions
                print("\nCalculating final evaluation metrics on the test set (Averaged Predictions)...")

                # Metrics for Post-processed Averaged Predictions
                mse_postprocessed = mean_squared_error(y_final_test_scaled, avg_final_test_predictions_scaled_postprocessed)
                rmse_postprocessed = np.sqrt(mse_postprocessed)
                mae_postprocessed = mean_absolute_error(y_final_test_scaled, avg_final_test_predictions_scaled_postprocessed)
                r2_postprocessed = r2_score(y_final_test_scaled, avg_final_test_predictions_scaled_postprocessed)

                epsilon = 1e-8
                mape_postprocessed = np.mean(np.abs((y_final_test_scaled - avg_final_test_predictions_scaled_postprocessed) / (y_final_test_scaled + epsilon))) * 100
                smape_postprocessed = np.mean(2.0 * np.abs(avg_final_test_predictions_scaled_postprocessed - y_final_test_scaled) / (np.abs(avg_final_test_predictions_scaled_postprocessed) + np.abs(y_final_test_scaled) + epsilon)) * 100


                print("\nModel Evaluation (Original Scale, Post-processed, Averaged):")
                print(f"Mean Squared Error (MSE): {mse_postprocessed:.2f}")
                print(f"Root Mean Squared Error (RMSE): {rmse_postprocessed:.2f}")
                print(f"Mean Absolute Error (MAE): {mae_postprocessed:.2f}")
                print(f"Mean Absolute Percentage Error (MAPE): {mape_postprocessed:.2f}%")
                print(f"Symmetric Mean Absolute Percentage Error (SMAPE): {smape_postprocessed:.2f}%")
                print(f"R² Score: {r2_postprocessed:.4f}")

                evaluation_results = {
                    'mse_postprocessed': mse_postprocessed,
                    'rmse_postprocessed': rmse_postprocessed,
                    'mae_postprocessed': mae_postprocessed,
                    'mape_postprocessed': mape_postprocessed,
                    'smape_postprocessed': smape_postprocessed,
                    'r2_postprocessed': r2_postprocessed,
                    'y_test_inv': scalers['target'].inverse_transform(y_final_test_scaled), # Store inverse transformed true test values
                    'y_pred_inv_postprocessed': avg_final_test_predictions_inv_postprocessed # Store averaged post-processed inverse predictions
                }

                # Metrics for Raw Averaged Predictions (Optional)
                try:
                    mse_raw = mean_squared_error(y_final_test_scaled, avg_final_test_predictions_scaled_raw)
                    rmse_raw = np.sqrt(mse_raw)
                    mae_raw = mean_absolute_error(y_final_test_scaled, avg_final_test_predictions_scaled_raw)
                    r2_raw = r2_score(y_final_test_scaled, avg_final_test_predictions_scaled_raw)

                    epsilon = 1e-8
                    mape_raw = np.mean(np.abs((y_final_test_scaled - avg_final_test_predictions_scaled_raw) / (y_final_test_scaled + epsilon))) * 100
                    smape_raw = np.mean(2.0 * np.abs(avg_final_test_predictions_scaled_raw - y_final_test_scaled) / (np.abs(avg_final_test_predictions_scaled_raw) + np.abs(y_final_test_scaled) + epsilon)) * 100

                    print("\nModel Evaluation (Original Scale, Raw, Averaged):")
                    print(f"Mean Squared Error (MSE): {mse_raw:.2f}")
                    print(f"Root Mean Squared Error (RMSE): {rmse_raw:.2f}")
                    print(f"Mean Absolute Error (MAE): {mae_raw:.2f}")
                    print(f"Mean Absolute Percentage Error (MAPE): {mape_raw:.2f}%")
                    print(f"Symmetric Mean Absolute Percentage Error (SMAPE): {smape_raw:.2f}%")
                    print(f"R² Score: {r2_raw:.4f}")

                    evaluation_results.update({
                        'mse_raw': mse_raw,
                        'rmse_raw': rmse_raw,
                        'mae_raw': mae_raw,
                        'mape_raw': mape_raw,
                        'smape_raw': smape_raw,
                        'r2_raw': r2_raw,
                        'y_pred_inv_raw': avg_final_test_predictions_inv_raw # Store averaged raw inverse predictions
                    })
                except Exception as e:
                     print(f"Warning: Could not calculate raw averaged metrics: {e}")

            else:
                print("\nNo successful folds trained for final evaluation. Cannot calculate metrics.")

        else: # Train a single model if n_splits_tscv is None or <= 1
            print("\nTraining a single final model on the entire training/validation set...")
            # Create sequences from the entire scaled training/validation data
            X_train_val_seq, y_train_val_seq = self.create_sequences(
                train_val_df_scaled[feature_info['all_features']],
                train_val_df_scaled[[feature_info['target']]], # Access scaled target
                self.sequence_length
            )

            # Ensure sequences are not empty
            if X_train_val_seq.shape[0] == 0:
                print("Error: Not enough data in the training/validation set to create sequences for final training. Exiting.")
                # Clear scaled train_val_df
                del train_val_df_scaled
                # Clear original final_test_df
                del final_test_df
                # Display plots generated so far (like TSCV splits if optimization is on)
                plt.show()
                return {} # Return empty results

            # Build final model with best (or default) hyperparameters
            input_shape = (X_train_val_seq.shape[1], X_train_val_seq.shape[2])
            model = self.build_model(input_shape)
            print("\nFinal Model Architecture (trained on full train/val):")
            model.summary()

            # Create callbacks for final training (no fold_idx needed for single model)
            callbacks = self.create_callbacks()

            # Train final model
            history = model.fit(
                X_train_val_seq, y_train_val_seq,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.15, # Use a fixed split for validation during final training for consistency
                callbacks=callbacks,
                verbose=1
            )
            last_history = history # Store history for plotting

            # Load the best model saved by the checkpoint callback during final training
            # Checkpoint path for single model is different
            best_model_path = os.path.join(self.model_dir, f'best_model_{self.timestamp}.keras')
            if os.path.exists(best_model_path):
                print(f"\nLoading best model from {best_model_path}")
                model = tf.keras.models.load_model(best_model_path, safe_mode=False, custom_objects={'clip_scaled_output': clip_scaled_output})
            else:
                print("\nWarning: Best model checkpoint not found for final training. Using the model from the last training epoch.")

            # Clear scaled train_val_df to free memory
            del train_val_df_scaled, X_train_val_seq, y_train_val_seq

            # --- Final Model Evaluation on Separate Test Set (Single Model) ---
            print("\nEvaluating final model on the separate test set...")
            # Scale the final test data using the scalers fitted on the train/val data
            final_test_df_scaled = self.scale_data_subset(final_test_df, feature_info, scalers)

            # Create sequences from the scaled final test data
            X_final_test_seq, y_final_test_scaled = self.create_sequences(
                 final_test_df_scaled[feature_info['all_features']],
                 final_test_df_scaled[[feature_info['target']]],
                 self.sequence_length
            )

            # Ensure final test sequences are not empty before proceeding
            if X_final_test_seq.shape[0] == 0:
                 print("Error: Not enough data in the final test set to create sequences for evaluation. Cannot perform final evaluation.")
                 # Clear scaled final_test_df
                 del final_test_df_scaled
                 # Clear original final_test_df
                 del final_test_df
                 # Display plots generated so far (like TSCV splits if optimization is on)
                 plt.show()
                 return {} # Return empty results

            # Slice the original final_test_df for post-processing logic, aligned with predictions
            original_final_test_df_slice_for_eval = final_test_df.iloc[self.sequence_length:].copy()

            # Make predictions on the final test set using the single trained model
            final_test_predictions_scaled_raw = model.predict(X_final_test_seq)
            final_test_predictions_scaled_raw = final_test_predictions_scaled_raw.reshape(-1, 1)

            # Apply post-processing to the predictions
            final_test_predictions_scaled_postprocessed = self.apply_zero_radiation_postprocessing(
                final_test_predictions_scaled_raw,
                original_final_test_df_slice_for_eval,
                self.radiation_column, # Pass radiation column name using class attribute
                scalers['target'] # Pass target scaler
            )

            # Calculate final evaluation metrics using the single model's predictions
            print("\nCalculating final evaluation metrics on the test set (Single Model)...")

            # Metrics for Post-processed Predictions
            mse_postprocessed = mean_squared_error(y_final_test_scaled, final_test_predictions_scaled_postprocessed)
            rmse_postprocessed = np.sqrt(mse_postprocessed)
            mae_postprocessed = mean_absolute_error(y_final_test_scaled, final_test_predictions_scaled_postprocessed)
            r2_postprocessed = r2_score(y_final_test_scaled, final_test_predictions_scaled_postprocessed)

            epsilon = 1e-8
            mape_postprocessed = np.mean(np.abs((y_final_test_scaled - final_test_predictions_scaled_postprocessed) / (y_final_test_scaled + epsilon))) * 100
            smape_postprocessed = np.mean(2.0 * np.abs(final_test_predictions_scaled_postprocessed - y_final_test_scaled) / (np.abs(final_test_predictions_scaled_postprocessed) + np.abs(y_final_test_scaled) + epsilon)) * 100


            print("\nModel Evaluation (Original Scale, Post-processed):")
            print(f"Mean Squared Error (MSE): {mse_postprocessed:.2f}")
            print(f"Root Mean Squared Error (RMSE): {rmse_postprocessed:.2f}")
            print(f"Mean Absolute Error (MAE): {mae_postprocessed:.2f}")
            print(f"Mean Absolute Percentage Error (MAPE): {mape_postprocessed:.2f}%")
            print(f"Symmetric Mean Absolute Percentage Error (SMAPE): {smape_postprocessed:.2f}%")
            print(f"R² Score: {r2_postprocessed:.4f}")

            evaluation_results = {
                'mse_postprocessed': mse_postprocessed,
                'rmse_postprocessed': rmse_postprocessed,
                'mae_postprocessed': mae_postprocessed,
                'mape_postprocessed': mape_postprocessed,
                'smape_postprocessed': smape_postprocessed,
                'r2_postprocessed': r2_postprocessed,
                'y_test_inv': scalers['target'].inverse_transform(y_final_test_scaled), # Store inverse transformed true test values
                'y_pred_inv_postprocessed': scalers['target'].inverse_transform(final_test_predictions_scaled_postprocessed) # Store post-processed inverse predictions
            }

            # Metrics for Raw Predictions (Optional)
            try:
                mse_raw = mean_squared_error(y_final_test_scaled, final_test_predictions_scaled_raw)
                rmse_raw = np.sqrt(rmse_raw)
                mae_raw = mean_absolute_error(y_test_inv, y_pred_inv_raw)
                r2_raw = r2_score(y_test_inv, y_pred_inv_raw)

                epsilon = 1e-8
                mape_raw = np.mean(np.abs((y_test_inv - y_pred_inv_raw) / (y_test_inv + epsilon))) * 100
                smape_raw = np.mean(2.0 * np.abs(y_pred_inv_raw - y_test_inv) / (np.abs(y_pred_inv_raw) + np.abs(y_test_inv) + epsilon)) * 100

                print("\nModel Evaluation (Original Scale, Raw):")
                print(f"Mean Squared Error (MSE): {mse_raw:.2f}")
                print(f"Root Mean Squared Error (RMSE): {rmse_raw:.2f}")
                print(f"Mean Absolute Error (MAE): {mae_raw:.2f}")
                print(f"Mean Absolute Percentage Error (MAPE): {mape_raw:.2f}%")
                print(f"Symmetric Mean Absolute Percentage Error (SMAPE): {smape_raw:.2f}%")
                print(f"R² Score: {r2_raw:.4f}")

                evaluation_results.update({
                    'mse_raw': mse_raw,
                    'rmse_raw': rmse_raw,
                    'mae_raw': mae_raw,
                    'mape_raw': mape_raw,
                    'smape_raw': smape_raw,
                    'r2_raw': r2_raw,
                    'y_pred_inv_raw': scalers['target'].inverse_transform(final_test_predictions_scaled_raw) # Store raw inverse predictions
                })
            except Exception as e:
                 print(f"Warning: Could not calculate raw metrics: {e}")

            # Clear scaled final_test_df and sequences
            del final_test_df_scaled, X_final_test_seq, y_final_test_scaled
            # Clear original final_test_df
            del final_test_df
            # Clear the single trained model
            del model
            tf.keras.backend.clear_session() # Clear backend session


        # --- Plotting and Saving Results ---
        # Note: Plotting functions now handle both averaged and single model results based on evaluation_results content
        if evaluation_results and last_history: # Only plot if evaluation was successful and history exists
            # Plot training history (from the last fold trained or single training)
            self.plot_results(last_history, evaluation_results)

            # Plot full test predictions vs actual (using averaged or single predictions)
            # Need to reload original_final_test_df for plotting index if needed
            # Or pass the index separately
            # For simplicity, let's pass the index from the original split
            # Ensure df_processed_before_split is available
            try:
                _, original_final_test_df_for_plotting = self.split_data_for_tscv_and_final_test(df_processed_before_split, final_test_size_ratio) # Reload
                self.plot_full_test_predictions(evaluation_results, original_final_test_df_for_plotting)
                del original_final_test_df_for_plotting # Clear after use
            except NameError:
                print("Warning: df_processed_before_split not defined. Cannot generate full test prediction plot.")


            # Capture model summary from one of the trained models (e.g., the last fold or the single model)
            # Need to build a model instance just to get the summary string
            temp_model_for_summary = self.build_model((self.sequence_length, len(feature_info['all_features'])))
            summary_string = []
            temp_model_for_summary.summary(print_fn=lambda x: summary_string.append(x))
            model_summary_str = "\n".join(summary_string)
            del temp_model_for_summary # Clear temp model

            # Save model summary
            self.save_model_summary(model_summary_str, feature_info, evaluation_results, data_path)
        else:
            print("\nSkipping plot and summary generation due to evaluation failure or missing history.")


        # --- Print final results ---
        print("\n--- Final Evaluation Results (Test Set) ---") # Reverted title
        if evaluation_results:
            # Print post-processed metrics if available, otherwise raw metrics (original then scaled fallback)
            if 'rmse_postprocessed' in evaluation_results:
                print(f"RMSE (Post-processed): {evaluation_results['rmse_postprocessed']:.2f}")
                print(f"MAE (Post-processed): {evaluation_results['mae_postprocessed']:.2f}")
                print(f"MAPE (Post-processed): {evaluation_results['mape_postprocessed']:.2f}%")
                print(f"SMAPE (Post-processed): {evaluation_results['smape_postprocessed']:.2f}%")
                print(f"R² (Post-processed): {evaluation_results['r2_postprocessed']:.4f}")
            elif 'rmse_raw' in evaluation_results: # Print raw metrics if calculated
                 print(f"RMSE (Raw): {evaluation_results['rmse_raw']:.2f}")
                 print(f"MAE (Raw): {evaluation_results['mae_raw']:.2f}")
                 print(f"MAPE (Raw): {evaluation_results['mape_raw']:.2f}%")
                 print(f"SMAPE (Raw): {evaluation_results['smape_raw']:.2f}%")
                 print(f"R² (Raw): {evaluation_results['r2_raw']:.4f}")
            else: # Fallback to scaled raw metrics if nothing else
                 print(f"RMSE (Scaled Raw Fallback): {evaluation_results['rmse_scaled']:.6f}")
                 print(f"MAE (Scaled Raw Fallback): {evaluation_results['mae_scaled']:.6f}")
                 print(f"MAPE (Scaled Raw Fallback): {evaluation_results['mape_scaled']:.2f}%")
                 print(f"SMAPE (Scaled Raw Fallback): {smape_scaled:.2f}%")
                 print(f"R² (Scaled Raw Fallback): {evaluation_results['r2_scaled']:.4f}")
        else:
            print("Evaluation could not be performed.")
        print("-" * 35)

        # Display all plots
        plt.show()

        return evaluation_results


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run LSTM model with Bayesian hyperparameter optimization and conditional TSCV for final training')
    parser.add_argument('--optimize', action='store_true', help='Perform hyperparameter optimization with TSCV')
    parser.add_argument('--sequence_length', type=int, default=24, help='Sequence length (hours to look back)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Maximum number of epochs for final model training')
    parser.add_argument('--trials', type=int, default=50, help='Number of Optuna optimization trials')
    # Changed default to None to differentiate between not parsed and parsed with value <= 1
    parser.add_argument('--n_splits_tscv', type=int, default=None, help='Number of splits for TimeSeriesSplit during optimization and final training. If not specified or <= 1, train a single model.')
    parser.add_argument('--final_test_size_ratio', type=float, default=0.15, help='Ratio of data to use for the final test set')
    # Updated default data path to the specified parquet file
    parser.add_argument('--data_path', type=str, default='data/processed/station_data_1h.parquet', help='Path to the data file')

    args = parser.parse_args()

    # Check for GPU availability
    print("TensorFlow version:", tf.__version__)
    gpu_devices = tf.config.list_physical_devices('GPU')
    print("GPU Available:", "Yes" if gpu_devices else "No")
    if not gpu_devices:
        print("Warning: No GPU devices found. Training may be very slow on CPU.")

    # Set memory growth to avoid memory allocation issues
    if gpu_devices:
        try:
            gpus = tf.config.list_physical_devices('GPU')
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
    N_SPLITS_TSCV_ARG = args.n_splits_tscv # Store the raw parsed argument value
    FINAL_TEST_SIZE_RATIO = args.final_test_size_ratio
    DATA_PATH = args.data_path

    print(f"Running LSTM model with parameters:")
    print(f"- Data path: {DATA_PATH}")
    print(f"- Data Resolution: 1 hour (as per data file)")
    print(f"- Sequence length: {SEQUENCE_LENGTH} steps ({SEQUENCE_LENGTH*1:.2f} hours lookback)")
    print(f"- Batch size: {BATCH_SIZE}")
    print(f"- Max epochs (final training): {EPOCHS}")
    print(f"- Hyperparameter optimization: {'Enabled' if OPTIMIZE else 'Disabled'}")
    if OPTIMIZE:
        print(f"- Number of optimization trials: {N_TRIALS}")

    # Determine the number of splits for final training based on the parsed argument
    if N_SPLITS_TSCV_ARG is not None and N_SPLITS_TSCV_ARG > 1:
        N_SPLITS_FINAL_TRAINING = N_SPLITS_TSCV_ARG
        print(f"- TSCV splits for final training: {N_SPLITS_FINAL_TRAINING}")
    else:
        N_SPLITS_FINAL_TRAINING = 1 # Use 1 split for single model training
        print("- Single model training for final evaluation (n_splits_tscv not specified or <= 1)")


    print(f"- Final test set ratio: {FINAL_TEST_SIZE_RATIO:.2f}")


    # Initialize and run forecaster
    forecaster = LSTMLowResForecaster(
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )

    # Run pipeline, passing the determined number of splits for final training
    metrics = forecaster.run_pipeline(
        data_path=DATA_PATH,
        optimize_hyperparams=OPTIMIZE,
        n_trials=N_TRIALS,
        n_splits_tscv=N_SPLITS_FINAL_TRAINING, # Pass the determined value here
        final_test_size_ratio=FINAL_TEST_SIZE_RATIO
    )

    # Final metrics are printed within run_pipeline now.
