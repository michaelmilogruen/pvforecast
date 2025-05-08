#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LSTM Model for PV Power Forecasting with High Resolution (10-minute) Data

This script implements an LSTM (Long Short-Term Memory) neural network for forecasting
photovoltaic power output using 10-minute resolution data. It includes data loading,
preprocessing, feature engineering, model training, and evaluation.

Key features:
1. Data preprocessing and feature scaling
2. Feature engineering for 10-minute resolution (including cyclical features)
3. LSTM model architecture
4. Hyperparameter optimization using Bayesian optimization with Optuna
5. Model evaluation and visualization
6. Fix for 'tf' not defined error in Lambda layer.
7. Modified to load data from a specified parquet file and use a specific list
   of features, including 'hour_cos' and 'isNight', without additional data augmentation.
8. Added function to plot actual vs predicted power for the entire test set.
9. Modified plotting functions to keep plots open after saving.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Lambda
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import optuna
from datetime import datetime
import argparse
import math # Import math for pi

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define the clipping function outside the class/method scope
# This ensures 'tf' is accessible when the function is called by the Lambda layer
def clip_scaled_output(x):
    """Clips the scaled output to be non-negative."""
    return tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=tf.float32.max)

class LSTMHighResForecaster:
    def __init__(self, sequence_length=24, batch_size=32, epochs=50):
        """
        Initialize the LSTM forecaster for high-resolution (10-minute) data.

        Args:
            sequence_length: Number of time steps (10-minute intervals) to look back (default: 144 for 24 hours)
            batch_size: Batch size for training
            epochs: Maximum number of epochs for training
        """
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.epochs = epochs

        # Create directories for model and results, specific to 10-minute data
        self.model_dir = 'models/lstm_10min'
        self.results_dir = 'results'
        self.optuna_results_dir = os.path.join(self.results_dir, 'lstm_10min_optuna')

        # Ensure directories exist
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.optuna_results_dir, exist_ok=True)


        # Timestamp for model versioning
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Default model configuration (will be updated by hyperparameter optimization)
        # Note: These ranges might need tuning based on the 10-minute data characteristics
        self.config = {
            'lstm_units': [48],  # Default 2 LSTM layers
            'dense_units': [24, 16],  # Default 2 dense layers
            'dropout_rates': [0.2, 0.2], # Adaptive dropout rates for LSTM layers
            'dense_dropout_rates': [0.05, 0.05], # Adaptive dropout rates for dense layers
            'learning_rate': 0.0005,
            'bidirectional': False,
            'batch_norm': True       }

    def load_data(self, data_path):
        """
        Load data for training from the specified file path.

        Args:
            data_path: Path to the data file (expected 10-minute resolution).

        Returns:
            Loaded dataframe with datetime index.
        """
        print(f"Loading data from {data_path}...")
        try:
            # Changed back to reading parquet file
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
        Create time-based and cyclical features from the datetime index for 10-minute data.
        Includes day of year cyclical features and isNight based on radiation.
        Note: 'hour_cos' is expected to be in the raw data based on user request.

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
        radiation_col_name = 'GlobalRadiation [W m-2]'
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
            'GlobalRadiation [W m-2]',
            'ClearSkyDHI',
            'ClearSkyGHI',
            'ClearSkyDNI',
            'SolarZenith [degrees]',
            'AOI [degrees]',
            'isNight', # Included as per user's list
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
            'GlobalRadiation [W m-2]',
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


        print(f"All features used ({len(features)}): {features}")
        print(f"MinMaxScaler features ({len(minmax_features)}): {minmax_features}")
        print(f"StandardScaler features ({len(standard_features)}): {standard_features}")
        print(f"RobustScaler features ({len(robust_features)}): {robust_features}")
        print(f"No scaling features ({len(no_scaling_features)}): {no_scaling_features}")


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

    def split_and_scale_data(self, df, feature_info):
        """
        Split data into train, validation, and test sets, then scale features.

        Args:
            df: DataFrame with all features and target
            feature_info: Dictionary with feature information

        Returns:
            Dictionary with split and scaled data and fitted scalers, plus original test_df
        """
        # Split data into train, validation, and test sets by time
        # Use 70% for training, 15% for validation, 15% for testing
        # Use iloc for position-based slicing after ensuring index is sorted
        df = df.sort_index() # Ensure data is sorted by time
        total_size = len(df)
        train_size = int(total_size * 0.7)
        val_size = int(total_size * 0.15)
        test_size = total_size - train_size - val_size # Use remaining for test

        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size : train_size + val_size]
        test_df = df.iloc[train_size + val_size : ].copy() # Keep original test_df and make a copy

        print(f"Total data size: {total_size}")
        print(f"Train set size: {len(train_df)}")
        print(f"Validation set size: {len(val_df)}")
        print(f"Test set size: {test_size}") # Corrected: Removed len() call


        # Initialize scalers for different feature groups
        # Only initialize if there are features in that group
        minmax_scaler = MinMaxScaler() if feature_info['minmax_features'] else None
        standard_scaler = StandardScaler() if feature_info['standard_features'] else None
        robust_scaler = RobustScaler() if feature_info['robust_features'] else None

        # Initialize target scaler (MinMaxScaler is common for output)
        target_scaler = MinMaxScaler(feature_range=(0, 1))

        # Fit scalers on training data only to prevent data leakage
        if minmax_scaler:
            minmax_scaler.fit(train_df[feature_info['minmax_features']])
            # Use os.path.join for robust path handling
            joblib.dump(minmax_scaler, os.path.join(self.model_dir, f'minmax_scaler_{self.timestamp}.pkl'))

        if standard_scaler:
            standard_scaler.fit(train_df[feature_info['standard_features']])
            joblib.dump(standard_scaler, os.path.join(self.model_dir, f'standard_scaler_{self.timestamp}.pkl'))

        if robust_scaler:
            robust_scaler.fit(train_df[feature_info['robust_features']])
            joblib.dump(robust_scaler, os.path.join(self.model_dir, f'robust_scaler_{self.timestamp}.pkl'))

        # Fit target scaler on training data
        target_scaler.fit(train_df[[feature_info['target']]])
        joblib.dump(target_scaler, os.path.join(self.model_dir, f'target_scaler_{self.timestamp}.pkl'))

        # Function to scale features using appropriate scalers
        def scale_features(df_subset):
            # Select only the features defined in feature_info['all_features']
            df_subset_selected = df_subset[feature_info['all_features']].copy()

            # Apply MinMaxScaler to appropriate features
            if minmax_scaler and feature_info['minmax_features']:
                df_subset_selected[feature_info['minmax_features']] = minmax_scaler.transform(df_subset_selected[feature_info['minmax_features']])

            # Apply StandardScaler to appropriate features
            if standard_scaler and feature_info['standard_features']:
                df_subset_selected[feature_info['standard_features']] = standard_scaler.transform(df_subset_selected[feature_info['standard_features']])

            # Apply RobustScaler to appropriate features
            if robust_scaler and feature_info['robust_features']:
                df_subset_selected[feature_info['robust_features']] = robust_scaler.transform(df_subset_selected[feature_info['robust_features']])

            # Features in no_scaling_features are already in df_subset_selected and not transformed

            return df_subset_selected

        # Apply scaling to each dataset
        X_train = scale_features(train_df)
        X_val = scale_features(val_df)
        X_test = scale_features(test_df) # Scale the features for creating sequences

        # Scale target variable
        y_train = target_scaler.transform(train_df[[feature_info['target']]])
        y_val = target_scaler.transform(val_df[[feature_info['target']]])
        y_test = target_scaler.transform(test_df[[feature_info['target']]]) # Scale the target for evaluation metrics


        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test, # Scaled y_test
            'original_test_df': test_df, # Return original test_df for post-processing checks and plotting index
            'scalers': {
                'minmax': minmax_scaler,
                'standard': standard_scaler,
                'robust': robust_scaler,
                'target': target_scaler
            },
            'feature_info': feature_info # Pass feature info to know radiation column name
        }

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

            # LSTM units for each layer
            lstm_units = []
            if num_lstm_layers >= 1:
                lstm_units.append(trial.suggest_int('lstm_units_1', 32, 64, step=8))
            if num_lstm_layers >= 2:
                lstm_units.append(trial.suggest_int('lstm_units_2', 16, 32, step=4))
            if num_lstm_layers >= 3:
                lstm_units.append(trial.suggest_int('lstm_units_3', 16, 128, step=8))
            if num_lstm_layers >= 4:
                lstm_units.append(trial.suggest_int('lstm_units_4', 8, 64, step=4))


            # Dense units for each layer
            dense_units = []
            if num_dense_layers >= 1:
                dense_units.append(trial.suggest_int('dense_units_1', 16, 32, step=4))
            if num_dense_layers >= 2:
                dense_units.append(trial.suggest_int('dense_units_2', 8, 16, step=2))
            if num_dense_layers >= 3:
                dense_units.append(trial.suggest_int('dense_units_3', 8, 32, step=4))

            # Dropout rates
            lstm_dropout_rates = []
            for i in range(num_lstm_layers):
                lstm_dropout_rates.append(trial.suggest_float(f'lstm_dropout_rate_{i+1}', 0.1, 0.5, step=0.05))

            dense_dropout_rates = []
            for i in range(num_dense_layers - 1): # Dropout between dense layers
                dense_dropout_rates.append(trial.suggest_float(f'dense_dropout_rate_{i+1}', 0.05, 0.3, step=0.05))

            learning_rate = trial.suggest_float('learning_rate', 5e-5, 5e-3, log=True)
            bidirectional = trial.suggest_categorical('bidirectional', [True, False])
            batch_norm = trial.suggest_categorical('batch_norm', [True, False])

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
            if i < num_dense_layers - 1:
                # Use dense dropout rate corresponding to this layer index or the last rate
                dense_dropout_rate_current = dense_dropout_rates[i] if i < len(dense_dropout_rates) else (dense_dropout_rates[-1] if dense_dropout_rates else 0.0)
                if dense_dropout_rate_current > 1e-6:
                    model.add(Dropout(dense_dropout_rate_current))

        # Output layer
        model.add(Dense(1))

        # --- Post-processing Layer(s) WITHIN the Keras model ---
        # This layer ensures the *scaled* output is not negative.
        # Using the defined function instead of a lambda
        model.add(Lambda(clip_scaled_output, name='scaled_output_clipping'))
        # --- End of Post-processing Layer(s) WITHIN the Keras model ---


        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mae', metrics=['mae', 'mse'])

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

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)

        if trial is None:
            model_checkpoint = ModelCheckpoint(
                filepath=os.path.join(self.model_dir, f'best_model_{self.timestamp}.keras'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
            callbacks.append(model_checkpoint)

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=0 if trial else 1
        )
        callbacks.append(reduce_lr)

        if trial:
            pruning_callback = optuna.integration.TFKerasPruningCallback(
                trial, 'val_loss'
            )
            callbacks.append(pruning_callback)

        return callbacks

    def objective(self, trial, X_train_seq, y_train_seq, X_val_seq, y_val_seq):
        """
        Objective function for Optuna optimization.
        """
        tf.keras.backend.clear_session()
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        model = self.build_model(input_shape, trial)
        callbacks = self.create_callbacks(trial)
        epochs_for_trial = 5

        try:
            history = model.fit(
                X_train_seq, y_train_seq,
                epochs=epochs_for_trial,
                batch_size=self.batch_size,
                validation_data=(X_val_seq, y_val_seq),
                callbacks=callbacks,
                verbose=0
            )
            validation_loss = min(history.history['val_loss'])
        except Exception as e:
            print(f"Trial {trial.number} failed due to error: {e}")
            raise optuna.exceptions.TrialPruned()

        return validation_loss

    def optimize_hyperparameters(self, X_train_seq, y_train_seq, X_val_seq, y_val_seq, n_trials=50):
        """
        Optimize hyperparameters using Optuna's Bayesian optimization.
        """
        print(f"\nStarting hyperparameter optimization with {n_trials} trials using Optuna...")
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
            interval_steps=1
        )
        db_path = os.path.join(self.optuna_results_dir, f'optuna_study_{self.timestamp}.db')
        study = optuna.create_study(
            direction='minimize',
            pruner=pruner,
            study_name=f'lstm_10min_study_{self.timestamp}',
            storage=f'sqlite:///{db_path}',
            load_if_exists=True
        )
        print(f"Optuna study stored at: {db_path}")
        if len(study.trials) > 0:
            print(f"Loaded existing study with {len(study.trials)} trials.")

        objective_func = lambda trial: self.objective(
            trial, X_train_seq, y_train_seq, X_val_seq, y_val_seq
        )

        remaining_trials = n_trials - len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        if remaining_trials > 0:
            print(f"Running {remaining_trials} new optimization trials...")
            study.optimize(objective_func, n_trials=remaining_trials, show_progress_bar=True)
        else:
            print("Optimization already completed for the specified number of trials.")

        if study.best_trial is None:
            print("No trials completed successfully.")
            return self.config
        else:
            best_params = study.best_params
            print("\nBest hyperparameters found:")
            for param, value in best_params.items():
                print(f"{param}: {value}")

            # Update config with best parameters
            num_lstm_layers = best_params.get('num_lstm_layers', len(self.config['lstm_units']))
            num_dense_layers = best_params.get('num_dense_layers', len(self.config['dense_units']))

            lstm_units = [best_params[f'lstm_units_{i+1}'] for i in range(num_lstm_layers) if f'lstm_units_{i+1}' in best_params]
            dense_units = [best_params[f'dense_units_{i+1}'] for i in range(num_dense_layers) if f'dense_units_{i+1}' in best_params]
            lstm_dropout_rates = [best_params[f'lstm_dropout_rate_{i+1}'] for i in range(num_lstm_layers) if f'lstm_dropout_rate_{i+1}' in best_params]
            dense_dropout_rates = [best_params[f'dense_dropout_rate_{i+1}'] for i in range(num_dense_layers - 1) if f'dense_dropout_rate_{i+1}'] # Note: dense_dropout_rate is for intermediate layers

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
                from optuna.visualization import plot_optimization_history, plot_param_importances
                print("Generating Optuna plots...")
                fig_history = plot_optimization_history(study)
                fig_history.write_image(os.path.join(self.optuna_results_dir, f'optimization_history_{self.timestamp}.png'))
                fig_importance = plot_param_importances(study)
                fig_importance.write_image(os.path.join(self.optuna_results_dir, f'param_importances_{self.timestamp}.png'))
                print("Optuna plots saved.")
            except ImportError:
                print("Warning: Optuna visualization dependencies (matplotlib, plotly) not found. Skipping plot generation.")
            except Exception as e:
                print(f"Warning: Could not generate Optuna plots. Error: {e}")


            best_params_path = os.path.join(self.model_dir, f'best_params_{self.timestamp}.txt')
            with open(best_params_path, 'w', encoding='utf-8') as f:
                f.write("Optimized Hyperparameters:\n")
                for param, value in self.config.items():
                    f.write(f"{param}: {value}\n")
            print(f"Best hyperparameters saved to {best_params_path}")

            return self.config

    def apply_zero_radiation_postprocessing(self, y_pred_scaled_raw: np.ndarray, original_test_df_slice: pd.DataFrame, radiation_column: str, target_scaler: MinMaxScaler) -> np.ndarray:
        """
        Applies post-processing: sets predicted power to 0 if future Global Radiation is 0.

        Args:
            y_pred_scaled_raw: Raw predictions from the model (scaled), shape (n_predictions, 1).
            original_test_df_slice: The original (unscaled) DataFrame slice used for the test set,
                                    starting from the first timestamp the model predicts for.
                                    Must be aligned with y_pred_scaled_raw.
            radiation_column: The name of the Global Radiation column in original_test_df_slice.
            target_scaler: The scaler used for the target variable, needed to transform 0 back to scaled space.

        Returns:
            Post-processed predictions (scaled), shape (n_predictions, 1).
        """
        print("\nApplying zero radiation post-processing...")

        if radiation_column not in original_test_df_slice.columns:
            print(f"Warning: Radiation column '{radiation_column}' not found in original test data for post-processing.")
            return y_pred_scaled_raw.copy() # Return a copy of raw predictions if radiation data is missing

        # Ensure lengths match - y_pred_scaled_raw should have the same number of predictions
        # as rows in original_test_df_slice if sliced correctly in evaluate_model.
        if len(y_pred_scaled_raw) != len(original_test_df_slice):
             print(f"Warning: Length mismatch between predictions ({len(y_pred_scaled_raw)}) and original test data slice ({len(original_test_df_slice)}). Skipping zero radiation post-processing.")
             return y_pred_scaled_raw.copy() # Return a copy of raw predictions

        # Create a boolean mask where Global Radiation is zero (or very close to zero)
        # at the prediction timestamps in the original test data.
        # Use a small threshold to account for floating point issues if necessary.
        RADIATION_THRESHOLD = 1.0 # W/m² - Adjust this threshold as needed
        zero_radiation_mask = (original_test_df_slice[radiation_column] < RADIATION_THRESHOLD)

        # Get the scaled value for zero power
        # This assumes the target scaler maps 0W to a specific scaled value (often 0)
        scaled_zero_value = target_scaler.transform([[0.0]])[0][0] # Transform 0W to scaled value

        # Apply the post-processing: set prediction to scaled_zero_value where radiation is zero
        y_pred_scaled_postprocessed = y_pred_scaled_raw.copy() # Work on a copy
        # Apply the mask. The mask needs to be reshaped to match the prediction shape (n_predictions, 1)
        y_pred_scaled_postprocessed[zero_radiation_mask.values.reshape(-1, 1)] = scaled_zero_value

        print(f"Applied zero radiation post-processing to {zero_radiation_mask.sum()} predictions.")

        return y_pred_scaled_postprocessed


    def evaluate_model(self, model, X_test_seq, y_test_scaled, original_test_df_slice, target_scaler, feature_info):
        """
        Evaluate the model and return metrics using post-processed predictions.

        Args:
            model: Trained LSTM model
            X_test_seq: Test sequences.
            y_test_scaled: True target values (scaled).
            original_test_df_slice: The original (unscaled) DataFrame slice corresponding to the predictions.
                                    Needed for post-processing rules based on original features.
                                    Must be aligned with y_pred_scaled_raw.
            target_scaler: Scaler for the target variable (needed for inverse transform).
            feature_info: Dictionary with feature information (to get radiation column name).


        Returns:
            Dictionary of evaluation metrics and actual/predicted values.
        """
        print("\nMaking predictions on the test set...")
        # Get raw scaled predictions from the model
        y_pred_scaled_raw = model.predict(X_test_seq)

        # Ensure y_test_scaled and y_pred_scaled_raw have the same shape (e.g., (n_samples, 1))
        y_test_scaled = y_test_scaled.reshape(-1, 1)
        y_pred_scaled_raw = y_pred_scaled_raw.reshape(-1, 1)

        # --- Apply Post-processing ---
        # Get the radiation column name from feature_info
        radiation_col_name = 'GlobalRadiation [W m-2]' # Assuming this is the correct name as used in features
        # You might want to verify this column is actually in original_test_df_slice

        y_pred_scaled_postprocessed = self.apply_zero_radiation_postprocessing(
            y_pred_scaled_raw,
            original_test_df_slice,
            radiation_col_name,
            target_scaler # Pass target_scaler to get the scaled zero value
        )
        # --- End of Post-processing ---


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

    def save_model_summary(self, model, feature_info, evaluation_results, data_path):
        """
        Saves a summary of the model architecture, configuration, and evaluation results.
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
            f.write("Model Configuration:\n")
            for key, value in self.config.items():
                f.write(f"    {key}: {value}\n")
            f.write("-" * 30 + "\n")
            f.write("Model Architecture:\n")
            model.summary(print_fn=lambda x: f.write(x + '\n'))
            f.write("-" * 30 + "\n")
            f.write("Evaluation Results (Test Set):\n")
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
            f.write(f"Best model saved to: {self.model_dir}/best_model_{self.timestamp}.keras\n")
            f.write(f"Results plots saved to: {self.results_dir}/lstm_10min_*{self.timestamp}.png\n")
            if os.path.exists(os.path.join(self.optuna_results_dir, f'optuna_study_{self.timestamp}.db')):
                 f.write(f"Optuna study database: {self.optuna_results_dir}/optuna_study_{self.timestamp}.db\n")
                 # Also list plot files if they were generated
                 if os.path.exists(os.path.join(self.optuna_results_dir, f'optimization_history_{self.timestamp}.png')):
                     f.write(f"Optuna plots saved to: {self.optuna_results_dir}/*_{self.timestamp}.png\n")


        print(f"\nModel summary saved to {summary_path}")


    def plot_results(self, history, evaluation_results):
        """
        Plot and save training history and sample prediction results.
        Plots post-processed predictions if available.

        Args:
            history: Training history object
            evaluation_results: Dictionary from model evaluation
        """
        print(f"Saving training history and sample prediction plots to {self.results_dir}/")

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
        plt.savefig(os.path.join(self.results_dir, f'lstm_10min_history_{self.timestamp}.png'))
        # plt.close() # Removed plt.close()


        # Determine which predictions to plot on scaled plots (post-processed is preferred)
        y_pred_scaled_to_plot = evaluation_results.get('y_pred_scaled_postprocessed', evaluation_results.get('y_pred_scaled_raw')) # Use get with default None or raw
        if y_pred_scaled_to_plot is None:
             print("Error: No scaled predictions available for plotting.")
             # Still try to plot scaled scatter if no inverse data at all
             if 'y_test_scaled' in evaluation_results: # Check if scaled data is available
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

                 plt.savefig(os.path.join(self.results_dir, f'lstm_10min_scatter_scaled_{self.timestamp}.png'))
                 # plt.close() # Removed plt.close()
             return # Exit plot function if no predictions

        # Define the suffix for the scaled plot title based on whether post-processing was applied
        plot_title_suffix_scaled = ' (Scaled, Post-processed)' if 'y_pred_scaled_postprocessed' in evaluation_results else ' (Scaled, Raw)'

        # Plot sample predictions vs actual (scaled)
        plt.figure(figsize=(14, 7))

        sample_size = min(1000, len(evaluation_results['y_test_scaled'])) # Reduced sample size for clarity
        indices = np.arange(sample_size)

        plt.plot(indices, evaluation_results['y_test_scaled'][:sample_size], 'b-', label='Actual (Scaled)')
        # Update label based on whether post-processing was applied
        pred_label_scaled = 'Predicted (Scaled, Post-processed)' if 'y_pred_scaled_postprocessed' in evaluation_results else 'Predicted (Scaled, Raw)'
        plt.plot(indices, y_pred_scaled_to_plot[:sample_size], 'r-', label=pred_label_scaled)
        plt.title(f'Actual vs Predicted PV Power{plot_title_suffix_scaled} - Sample ({sample_size} points)')
        plt.xlabel(f'Time Steps ({self.sequence_length}-step lookback)')
        plt.ylabel('Power Output (Scaled)')
        plt.legend()
        plt.grid(True)

        plt.savefig(os.path.join(self.results_dir, f'lstm_10min_predictions_scaled_sample_{self.timestamp}.png'))
        # plt.close() # Removed plt.close()


        # Determine which inverse-transformed predictions to plot (post-processed is preferred)
        y_pred_inv_to_plot = evaluation_results.get('y_pred_inv_postprocessed', evaluation_results.get('y_pred_inv_raw')) # Use get with default None or raw inverse
        if y_pred_inv_to_plot is None:
             print("Warning: No inverse-transformed predictions available for plotting on original scale.")
             return # Exit plot function if no inverse transformed data

        # Plot sample predictions vs actual (original scale)
        plt.figure(figsize=(14, 7))

        sample_size_inv = min(1000, len(evaluation_results['y_test_inv'])) # Reduced sample size for clarity
        indices_inv = np.arange(sample_size_inv)

        plt.plot(indices_inv, evaluation_results['y_test_inv'][:sample_size_inv], 'b-', label='Actual Power Output')
        # Update label based on whether post-processing was applied
        pred_label_inv = 'Predicted Power Output (Post-processed)' if 'y_pred_inv_postprocessed' in evaluation_results else 'Predicted Power Output (Raw)'
        plt.plot(indices_inv, y_pred_inv_to_plot[:sample_size_inv], 'r-', label=pred_label_inv)
        plt.title(f'Actual vs Predicted PV Power (W) - Sample ({sample_size_inv} points)')
        plt.xlabel(f'Time Steps ({self.sequence_length}-step lookback)')
        plt.ylabel('Power Output (W)')
        plt.legend()
        plt.grid(True)

        plt.savefig(os.path.join(self.results_dir, f'lstm_10min_predictions_sample_{self.timestamp}.png'))
        # plt.close() # Removed plt.close()

        # Create scatter plot (original scale) - Sampled for performance
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
        plt.savefig(os.path.join(self.results_dir, f'lstm_10min_scatter_{self.timestamp}.png'))
        # plt.close() # Removed plt.close()

        print("Sample plots saved.")

    def plot_full_test_predictions(self, evaluation_results, original_test_df):
        """
        Plots the actual vs predicted power output for the entire test dataset.

        Args:
            evaluation_results: Dictionary containing 'y_test_inv' and
                                'y_pred_inv_postprocessed' (or 'y_pred_inv_raw').
            original_test_df: The original (unscaled) DataFrame for the test set.
                              Used to get the datetime index for plotting.
                              Must be the full test set DataFrame returned by split_and_scale_data.
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
        prediction_index = original_test_df.index[self.sequence_length:]

        # Ensure the index length matches the prediction length
        if len(prediction_index) != len(y_test_inv):
            print(f"Warning: Index length ({len(prediction_index)}) does not match prediction length ({len(y_test_inv)}). Cannot plot full test predictions.")
            return

        plt.figure(figsize=(18, 8)) # Larger figure for the full test set
        plt.plot(prediction_index, y_test_inv, label='Actual Power Output')

        # Update label based on whether post-processing was applied
        pred_label_inv = 'Predicted Power Output (Post-processed)' if 'y_pred_inv_postprocessed' in evaluation_results else 'Predicted Power Output (Raw)'
        plt.plot(prediction_index, y_pred_inv_to_plot, label=pred_label_inv, alpha=0.7) # Use alpha for potentially dense plots

        plt.title('Actual vs Predicted PV Power (W) - Full Test Set')
        plt.xlabel('Time')
        plt.ylabel('Power Output (W)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45) # Rotate x-axis labels for better readability
        plt.tight_layout() # Adjust layout to prevent labels overlapping

        plot_path = os.path.join(self.results_dir, f'lstm_10min_predictions_full_test_{self.timestamp}.png')
        plt.savefig(plot_path)
        # plt.close() # Removed plt.close()

        print(f"Full test prediction plot saved to {plot_path}")


    def run_pipeline(self, data_path, optimize_hyperparams=True, n_trials=50):
        """
        Run the full LSTM forecasting pipeline for high-resolution (10-minute) data.

        Args:
            data_path: Path to the 10-minute resolution data file.
            optimize_hyperparams: Whether to perform hyperparameter optimization.
            n_trials: Number of optimization trials.

        Returns:
            Dictionary of evaluation metrics.
        """
        print(f"Running LSTM forecasting pipeline for high-resolution (10-minute) data from {data_path}")

        # --- Data Loading and Preparation ---
        # Load the raw data
        df = self.load_data(data_path)

        # Create time features (day_sin, day_cos, isNight)
        # Note: hour_cos is expected to be in the raw data as per user request
        df = self.create_time_features(df)

        # Prepare features (defines which columns to use and how to scale)
        # This step now uses the specific list of features requested by the user
        # plus the engineered day features and isNight.
        feature_info = self.prepare_features(df)

        # Select only the features and target defined in prepare_features before splitting/scaling
        # This ensures we are working only with the specified columns.
        required_cols = feature_info['all_features'] + [feature_info['target']]
        # Check if all required columns exist after loading and feature engineering
        missing_required_cols = [col for col in required_cols if col not in df.columns]
        if missing_required_cols:
            print(f"Error: The following required columns (features or target) are missing before splitting/scaling: {missing_required_cols}. Exiting.")
            exit()

        df_processed = df[required_cols].copy()


        # Split and scale data (applies scalers and creates splits, returns original test_df)
        # The original_test_df is needed for post-processing checks based on original feature values
        data = self.split_and_scale_data(df_processed, feature_info) # Corrected method name

        # Clear dataframe after splitting/scaling to free memory
        del df, df_processed
        # --- End of Data Loading and Preparation ---


        # Create sequences for model input
        print(f"\nCreating sequences with sequence length: {self.sequence_length} steps ({self.sequence_length*10/60:.2f} hours lookback)")

        X_train_seq, y_train_seq = self.create_sequences(
            data['X_train'], data['y_train'], self.sequence_length
        )
        X_val_seq, y_val_seq = self.create_sequences(
            data['X_val'], data['y_val'], self.sequence_length
        )
        # Note: X_test_seq and y_test_scaled are created for model input and evaluation
        # y_test_scaled here represents the true target values, scaled.
        X_test_seq, y_test_scaled = self.create_sequences(
            data['X_test'], data['y_test'], self.sequence_length
        )

        print(f"Training sequences shape: {X_train_seq.shape}")
        print(f"Validation sequences shape: {X_val_seq.shape}")
        print(f"Testing sequences shape: {X_test_seq.shape}")


        # Perform hyperparameter optimization if requested
        if optimize_hyperparams:
            print("\nPerforming hyperparameter optimization...")
            self.optimize_hyperparameters(
                X_train_seq, y_train_seq, X_val_seq, y_val_seq, n_trials=n_trials
            )
            print(f"\nOptimized hyperparameters loaded into configuration.")
        else:
            print("\nHyperparameter optimization skipped. Using default or previously loaded configuration.")


        # Build final model
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        model = self.build_model(input_shape)
        print("\nFinal Model Architecture:")
        model.summary()

        # Create callbacks for final training
        callbacks = self.create_callbacks()

        # Train model
        print("\nTraining final model...")
        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val_seq, y_val_seq),
            callbacks=callbacks,
            verbose=1
        )

        # Load the best model saved by the checkpoint callback
        best_model_path = os.path.join(self.model_dir, f'best_model_{self.timestamp}.keras')
        if os.path.exists(best_model_path):
            print(f"\nLoading best model from {best_model_path}")
            # Pass custom_objects when loading the model to ensure the Lambda layer is handled correctly
            model = tf.keras.models.load_model(best_model_path, safe_mode=False, custom_objects={'clip_scaled_output': clip_scaled_output})
        else:
            print("\nWarning: Best model checkpoint not found. Using the model from the last training epoch.")


        # Evaluate model using the original test_df slice corresponding to the predictions
        # Need to slice the original_test_df from data['original_test_df']
        # The predictions y_pred will correspond to the timestamps starting at index self.sequence_length
        original_test_df_slice_for_eval = data['original_test_df'].iloc[self.sequence_length:].copy()

        print("\nEvaluating model using post-processing...")
        evaluation_results = self.evaluate_model(
            model,
            X_test_seq,
            y_test_scaled, # Pass scaled true targets
            original_test_df_slice_for_eval, # Pass original test data slice for post-processing logic
            data['scalers']['target'], # Pass target scaler for inverse transform and scaled zero value
            data['feature_info'] # Pass feature_info to know radiation column name
        )

        # Plot training history and sample results
        self.plot_results(history, evaluation_results)

        # Plot full test predictions vs actual
        # Pass the *full* original test DataFrame to the new plotting function
        self.plot_full_test_predictions(evaluation_results, data['original_test_df'])


        # Save model summary
        self.save_model_summary(model, data['feature_info'], evaluation_results, data_path)


        # --- Print final results ---
        print("\n--- Final Evaluation Results (Test Set) ---")
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
             print(f"SMAPE (Scaled Raw Fallback): {evaluation_results['smape_scaled']:.2f}%")
             f.write(f"R² (Scaled Raw Fallback): {evaluation_results['r2_scaled']:.4f}\n")


        print("-" * 35)
        print(f"Model artifacts saved to {self.model_dir}/")
        print(f"Results plots saved to {self.results_dir}/")
        if optimize_hyperparams:
            print(f"Optuna results saved to {self.optuna_results_dir}/")

        # Display all plots
        plt.show()

        return evaluation_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run LSTM model for 10-minute PV forecasting with Bayesian hyperparameter optimization')
    parser.add_argument('--optimize', action='store_true', help='Perform hyperparameter optimization')
    parser.add_argument('--sequence_length', type=int, default=144, help='Sequence length (10-minute intervals to look back, default 144 for 24 hours)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (adjust based on GPU memory)')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of epochs for final model training')
    parser.add_argument('--trials', type=int, default=50, help='Number of Optuna optimization trials')
    # Updated default data path to the specified parquet file
    parser.add_argument('--data_path', type=str, default='data/processed/station_data_10min.parquet', help='Path to the 10-minute resolution data file')

    args = parser.parse_args()

    print("TensorFlow version:", tf.__version__)
    gpu_devices = tf.config.list_physical_devices('GPU')
    print("GPU Available:", "Yes" if gpu_devices else "No")
    if not gpu_devices:
        print("Warning: No GPU devices found. Training may be very slow on CPU.")

    if gpu_devices:
        try:
            # Corrected: Access the list of GPUs correctly
            gpus = tf.config.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory growth set to True for all GPUs")
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")

    SEQUENCE_LENGTH = args.sequence_length
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    OPTIMIZE = args.optimize
    N_TRIALS = args.trials
    DATA_PATH = args.data_path # This will now be the path to your market file

    print(f"\nRunning LSTM model with parameters:")
    print(f"- Data path: {DATA_PATH}")
    print(f"- Data Resolution: 10 minutes")
    print(f"- Sequence length: {SEQUENCE_LENGTH} steps ({SEQUENCE_LENGTH*10/60:.2f} hours lookback)")
    print(f"- Batch size: {BATCH_SIZE}")
    print(f"- Max epochs (final training): {EPOCHS}")
    print(f"- Hyperparameter optimization: {'Enabled' if OPTIMIZE else 'Disabled'}")
    if OPTIMIZE:
        print(f"- Number of optimization trials: {N_TRIALS}")

    forecaster = LSTMHighResForecaster(
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )

    # Call the main pipeline method
    metrics = forecaster.run_pipeline(
        data_path=DATA_PATH,
        optimize_hyperparams=OPTIMIZE,
        n_trials=N_TRIALS
    )
