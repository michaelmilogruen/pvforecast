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
10. Adjusted Optuna search space based on provided good parameters.
11. Increased epochs per Optuna trial.
12. Added Optuna visualization plots (intermediate values, parameter importances,
    optimization history, parallel coordinate, and slice plots).
13. Added command-line arguments for --optimize and --batch_size.
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
import optuna.visualization # Import Optuna visualization module
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
    def __init__(self, sequence_length=144, batch_size=32, epochs=50): # Default sequence_length to 144 (24 hours of 10-min data)
        """
        Initialize the LSTM forecaster for high-resolution (10-minute) data.

        Args:
            sequence_length: Number of time steps (10-minute intervals) to look back (default: 144 for 24 hours)
            batch_size: Batch size for training
            epochs: Maximum number of epochs for training (for final model training, not Optuna trials)
        """
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.epochs = epochs # This is for the final training, not Optuna trials

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
        # These defaults are based on the provided 'good parameters'
        self.config = {
            'num_lstm_layers': 1, # Based on good parameters
            'lstm_units': [64], # Based on good parameters (64 per direction for Bidirectional)
            'num_dense_layers': 1, # Number of HIDDEN dense layers (good params had 1 hidden + 1 output)
            'dense_units': [24], # Units for the first hidden dense layer
            'dropout_rates': [0.3], # Example dropout after LSTM
            'dense_dropout_rates': [0.1], # Example dropout after hidden dense
            'learning_rate': 0.001, # A reasonable default
            'bidirectional': True, # Based on good parameters
            'batch_norm': True # Based on good parameters
        }

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
        print(f"Test set size: {len(test_df)}") # Corrected: Use len(test_df)


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
            # --- Optuna Hyperparameter Suggestions (Adjusted based on good parameters) ---

            # Number of LSTM layers: Good params suggest 1 Bidirectional layer. Explore 1 or 2.
            num_lstm_layers = trial.suggest_int('num_lstm_layers', 1, 2)

            # LSTM units: Good params suggest ~64 per direction for the first layer.
            # Narrow the search space around 64 for the first layer.
            lstm_units = []
            if num_lstm_layers >= 1:
                lstm_units.append(trial.suggest_int('lstm_units_1', 48, 80, step=8)) # Narrowed range around 64
            if num_lstm_layers >= 2:
                # If a second layer exists, search a lower range
                lstm_units.append(trial.suggest_int('lstm_units_2', 16, 48, step=8)) # Adjusted range

            # Number of HIDDEN Dense layers: Good params suggest 1 hidden dense layer (plus output). Explore 0 or 1 hidden.
            num_dense_layers = trial.suggest_int('num_hidden_dense_layers', 0, 1)

            # Dense units: Good params suggest 24 for the first hidden layer.
            dense_units = []
            if num_dense_layers >= 1:
                    # Only suggest units for the first hidden layer if num_hidden_dense_layers is at least 1
                    dense_units.append(trial.suggest_int('dense_units_1', 16, 32, step=4)) # Narrowed range around 24
            # Note: The output layer (1 unit) is added separately and its size is not a hyperparameter here.


            # Dropout rates: Keep a reasonable range, good params used dropout.
            lstm_dropout_rates = []
            for i in range(num_lstm_layers):
                lstm_dropout_rates.append(trial.suggest_float(f'lstm_dropout_rate_{i+1}', 0.1, 0.4, step=0.05)) # Slightly narrowed range

            dense_dropout_rates = []
            # Add dropout rate suggestion only if there's at least one hidden dense layer
            if num_dense_layers >= 1:
                    dense_dropout_rates.append(trial.suggest_float('dense_dropout_rate_1', 0.05, 0.25, step=0.05)) # Slightly narrowed range


            # Learning rate: Keep a broad log-uniform range.
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True) # Adjusted log range

            # Bidirectional: Good params used Bidirectional. Keep as categorical.
            bidirectional = trial.suggest_categorical('bidirectional', [True, False])

            # Batch Norm: Good params used Batch Norm. Keep as categorical.
            batch_norm = trial.suggest_categorical('batch_norm', [True, False])

            # Update self.config with suggested parameters for tracking (optional, but good practice)
            self.config['num_lstm_layers'] = num_lstm_layers
            self.config['lstm_units'] = lstm_units
            self.config['num_dense_layers'] = num_dense_layers # Store number of hidden dense layers
            self.config['dense_units'] = dense_units
            self.config['dropout_rates'] = lstm_dropout_rates
            self.config['dense_dropout_rates'] = dense_dropout_rates
            self.config['learning_rate'] = learning_rate
            self.config['bidirectional'] = bidirectional
            self.config['batch_norm'] = batch_norm

        # Retrieve hyperparameters from self.config (either defaults or from trial)
        num_lstm_layers = self.config['num_lstm_layers']
        lstm_units = self.config['lstm_units']
        num_hidden_dense_layers = self.config['num_dense_layers'] # Use the stored number of hidden layers
        dense_units = self.config['dense_units']
        dropout_rates = self.config['dropout_rates']
        dense_dropout_rates = self.config['dense_dropout_rates']
        learning_rate = self.config['learning_rate']
        bidirectional = self.config['bidirectional']
        batch_norm = self.config['batch_norm']


        model = Sequential()

        # Add LSTM layers
        for i in range(num_lstm_layers):
            is_last_lstm = (i == num_lstm_layers - 1)
            # return_sequences should be True for all but the last LSTM layer
            return_sequences = not is_last_lstm

            lstm_layer_instance = LSTM(units=lstm_units[i], return_sequences=return_sequences)

            if i == 0: # First LSTM layer needs input_shape
                if bidirectional:
                    model.add(Bidirectional(lstm_layer_instance, input_shape=input_shape))
                else:
                    model.add(LSTM(units=lstm_units[i], return_sequences=return_sequences, input_shape=input_shape))
            else: # Subsequent LSTM layers
                if bidirectional:
                    model.add(Bidirectional(lstm_layer_instance))
                else:
                    model.add(lstm_layer_instance)

            if batch_norm:
                model.add(BatchNormalization())
            # Add dropout after each LSTM layer (if rate is > 0)
            dropout_rate_lstm = dropout_rates[i] if i < len(dropout_rates) else 0.0
            if dropout_rate_lstm > 1e-6:
                model.add(Dropout(dropout_rate_lstm))


        # Add HIDDEN Dense layers
        for i in range(num_hidden_dense_layers):
            model.add(Dense(dense_units[i], activation='relu'))
            if batch_norm:
                model.add(BatchNormalization())
            # Add dropout after hidden dense layers (if rate is > 0)
            dropout_rate_dense = dense_dropout_rates[i] if i < len(dense_dropout_rates) else 0.0
            if dropout_rate_dense > 1e-6:
                    model.add(Dropout(dropout_rate_dense))


        # Output layer (always 1 unit, no activation for regression)
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
            patience=10, # Increased patience for Optuna trials as well
            restore_best_weights=True
        )
        callbacks.append(early_stopping)

        if trial is None: # Only save checkpoint for the final best model training
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
            # Optuna Pruning Callback
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

        # Increased epochs for each Optuna trial
        epochs_for_trial = 30 # Increased from 5

        try:
            history = model.fit(
                X_train_seq, y_train_seq,
                epochs=epochs_for_trial,
                batch_size=self.batch_size, # Use the forecaster's batch size
                validation_data=(X_val_seq, y_val_seq),
                callbacks=callbacks,
                verbose=0 # Keep verbose low during trials
            )
            # Return the minimum validation loss achieved during the trial
            validation_loss = min(history.history['val_loss'])
        except Exception as e:
            print(f"Trial {trial.number} failed due to error: {e}")
            # Optionally log the error or trial parameters for debugging
            # trial.set_user_attr("error", str(e))
            raise optuna.exceptions.TrialPruned() # Prune trial on error

        return validation_loss

    def optimize_hyperparameters(self, X_train_seq, y_train_seq, X_val_seq, y_val_seq, n_trials=50):
        """
        Optimize hyperparameters using Optuna's Bayesian optimization.
        Includes saving visualization plots.
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
            completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            print(f"Completed trials in loaded study: {completed_trials}")


        objective_func = lambda trial: self.objective(
            trial, X_train_seq, y_train_seq, X_val_seq, y_val_seq
        )

        # Calculate remaining trials to run
        completed_trials_count = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        remaining_trials = max(0, n_trials - completed_trials_count)


        if remaining_trials > 0:
            print(f"Running {remaining_trials} new optimization trials...")
            study.optimize(objective_func, n_trials=remaining_trials, show_progress_bar=True)
        else:
            print("Optimization already completed for the specified number of trials or no new trials requested.")

        if study.best_trial is None:
            print("No trials completed successfully during optimization.")
            # Optionally, if loading an existing study, check if it has best trials
            if completed_trials_count > 0:
                print(f"Using parameters from the best trial ({study.best_value:.4f}) from the loaded study.")
                best_params = study.best_params
            else:
                print("No completed trials found in the study. Using default configuration.")
                # Return default config if no successful trials
                return self.config
        else:
            best_params = study.best_params
            print("\nBest hyperparameters found:")
            for param, value in best_params.items():
                print(f"{param}: {value}")

        # --- Generate and Save Optuna Visualization Plots ---
        print("Generating Optuna visualization plots...")
        try:
            # Ensure plotting dependencies are installed
            from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate, plot_slice, plot_intermediate_values

            # Plot intermediate values (optimization history during trials)
            fig_intermediate = plot_intermediate_values(study)
            fig_intermediate.write_html(os.path.join(self.optuna_results_dir, f'optuna_intermediate_values_{self.timestamp}.html'))
            print(f"Saved intermediate values plot to {self.optuna_results_dir}")

            # Plot parameter importances
            fig_importance = plot_param_importances(study)
            fig_importance.write_html(os.path.join(self.optuna_results_dir, f'optuna_parameter_importances_{self.timestamp}.html'))
            print(f"Saved parameter importances plot to {self.optuna_results_dir}")

            # Plot optimization history (overall study history)
            fig_history = plot_optimization_history(study)
            fig_history.write_html(os.path.join(self.optuna_results_dir, f'optuna_optimization_history_{self.timestamp}.html'))
            print(f"Saved optimization history plot to {self.optuna_results_dir}")

            # Plot high-dimensional parameter relationships (if applicable)
            # This plot can be slow for many parameters, so it's optional
            try:
                 fig_parallel_coordinate = plot_parallel_coordinate(study)
                 fig_parallel_coordinate.write_html(os.path.join(self.optuna_results_dir, f'optuna_parallel_coordinate_{self.timestamp}.html'))
                 print(f"Saved parallel coordinate plot to {self.optuna_results_dir}")
            except Exception as e:
                 print(f"Could not generate parallel coordinate plot (requires more than 1 trial with diverse parameters): {e}")

            # Plot slice plots for key hyperparameters that exist in the study
            # Get all parameter names from the study
            study_params = list(study.best_params.keys())
            key_params_to_plot = ['num_lstm_layers', 'num_hidden_dense_layers', 'learning_rate'] + [p for p in study_params if 'units' in p or 'dropout_rate' in p]
            key_params_to_plot = list(dict.fromkeys(key_params_to_plot)) # Remove duplicates

            for param in key_params_to_plot:
                if param in study_params:
                    try:
                        # Optuna slice plots use matplotlib internally, need to handle figure
                        plt.figure(figsize=(10, 6))
                        plot_slice(study, params=[param])
                        plt.tight_layout()
                        plt.savefig(os.path.join(self.optuna_results_dir, f'optuna_slice_{param}_{self.timestamp}.png'))
                        plt.close() # Close figure after saving
                    except Exception as e_plot:
                         print(f"Warning: Could not generate slice plot for {param}. Error: {e_plot}")

            print(f"Optuna visualization plots saved to {self.optuna_results_dir}/")

        except ImportError:
            print("Warning: Optuna visualization dependencies (plotly, kaleido, matplotlib) not found. Skipping plot generation.")
            print("Install them with: pip install plotly kaleido matplotlib")
        except Exception as e:
            print(f"Error generating Optuna plots: {e}")


        # Update config with best parameters found
        # Ensure keys match the trial.suggest names
        self.config['num_lstm_layers'] = best_params.get('num_lstm_layers', self.config['num_lstm_layers'])
        # Reconstruct lstm_units list based on best_params and num_lstm_layers
        best_lstm_units = []
        for i in range(self.config['num_lstm_layers']):
            unit_key = f'lstm_units_{i+1}'
            if unit_key in best_params:
                best_lstm_units.append(best_params[unit_key])
            elif i < len(self.config['lstm_units']):
                    best_lstm_units.append(self.config['lstm_units'][i]) # Fallback to default if key missing
        self.config['lstm_units'] = best_lstm_units


        self.config['num_dense_layers'] = best_params.get('num_hidden_dense_layers', self.config['num_dense_layers']) # Update number of hidden layers
        # Reconstruct dense_units list based on best_params and num_hidden_dense_layers
        best_dense_units = []
        for i in range(self.config['num_dense_layers']):
            unit_key = f'dense_units_{i+1}'
            if unit_key in best_params:
                best_dense_units.append(best_params[unit_key])
            elif i < len(self.config['dense_units']):
                    best_dense_units.append(self.config['dense_units'][i]) # Fallback to default
        self.config['dense_units'] = best_dense_units


        # Reconstruct dropout lists similarly
        best_lstm_dropout_rates = []
        for i in range(self.config['num_lstm_layers']):
             rate_key = f'lstm_dropout_rate_{i+1}'
             if rate_key in best_params:
                 best_lstm_dropout_rates.append(best_params[rate_key])
             elif i < len(self.config['dropout_rates']):
                 best_lstm_dropout_rates.append(self.config['dropout_rates'][i])
        self.config['dropout_rates'] = best_lstm_dropout_rates

        best_dense_dropout_rates = []
        for i in range(self.config['num_dense_layers']): # Dropout applies after hidden layers
             rate_key = f'dense_dropout_rate_{i+1}'
             if rate_key in best_params:
                 best_dense_dropout_rates.append(best_params[rate_key])
             elif i < len(self.config['dense_dropout_rates']):
                 best_dense_dropout_rates.append(self.config['dense_dropout_rates'][i])
        self.config['dense_dropout_rates'] = best_dense_dropout_rates


        self.config['learning_rate'] = best_params.get('learning_rate', self.config['learning_rate'])
        self.config['bidirectional'] = best_params.get('bidirectional', self.config['bidirectional'])
        self.config['batch_norm'] = best_params.get('batch_norm', self.config['batch_norm'])


        print("\nUpdated configuration with best parameters:")
        print(self.config)

        # Save optimized hyperparameters to a text file
        best_params_path = os.path.join(self.model_dir, f'best_params_{self.timestamp}.txt')
        with open(best_params_path, 'w', encoding='utf-8') as f:
            f.write("Optimized Hyperparameters:\n")
            for param, value in self.config.items():
                f.write(f"  {param}: {value}\n")
        print(f"Best hyperparameters saved to {best_params_path}")

        # Save study for later analysis
        study_save_path = os.path.join(self.optuna_results_dir, f'study_{self.timestamp}.pkl')
        joblib.dump(study, study_save_path)
        print(f"Optuna study object saved to {study_save_path}")


        return self.config # Return the best configuration


    def train_final_model(self, X_train_seq, y_train_seq, X_val_seq, y_val_seq, best_config):
        """
        Train the final model using the best hyperparameters found by Optuna.
        """
        print("\nTraining final model with best hyperparameters...")
        tf.keras.backend.clear_session()
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])

        # Ensure self.config is updated with the best_config before building the model
        self.config.update(best_config)

        model = self.build_model(input_shape) # Build model using the updated self.config
        callbacks = self.create_callbacks() # Create callbacks for final training (includes ModelCheckpoint)

        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=self.epochs, # Use the full number of epochs specified for final training
            batch_size=self.batch_size,
            validation_data=(X_val_seq, y_val_seq),
            callbacks=callbacks,
            verbose=1
        )

        # Save the final trained model
        final_model_path = os.path.join(self.model_dir, f'final_model_{self.timestamp}.keras')
        model.save(final_model_path)
        print(f"Final model saved to {final_model_path}")

        return model, history


    def evaluate_model(self, model, X_test_seq, y_test_scaled, original_test_df, target_scaler):
        """
        Evaluate the trained model on the test set and inverse scale predictions.
        """
        print("\nEvaluating model on test set...")
        # Predict on the test set (scaled output)
        y_pred_scaled = model.predict(X_test_seq)

        # Inverse scale predictions and actual values
        y_pred = target_scaler.inverse_transform(y_pred_scaled)
        y_actual = target_scaler.inverse_transform(y_test_scaled)

        # Ensure original_test_df has the correct index for plotting
        # The sequences are created from index i to i + time_steps, predicting i + time_steps
        # So the predictions correspond to original_test_df starting from index = self.sequence_length
        test_index_for_predictions = original_test_df.index[self.sequence_length:]

        # Align predictions and actuals with the correct index from original_test_df
        # This is crucial for plotting and time-based analysis
        y_pred_aligned = pd.Series(y_pred.flatten(), index=test_index_for_predictions)
        y_actual_aligned = pd.Series(y_actual.flatten(), index=test_index_for_predictions)

        # Calculate metrics using inverse-scaled values
        mae = mean_absolute_error(y_actual_aligned, y_pred_aligned)
        rmse = np.sqrt(mean_squared_error(y_actual_aligned, y_pred_aligned))
        r2 = r2_score(y_actual_aligned, y_pred_aligned)

        print(f"Test MAE: {mae:.4f}")
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test R2: {r2:.4f}")

        # Save metrics
        metrics_path = os.path.join(self.results_dir, f'test_metrics_{self.timestamp}.txt')
        with open(metrics_path, 'w') as f:
            f.write(f"Test MAE: {mae:.4f}\n")
            f.write(f"Test RMSE: {rmse:.4f}\n")
            f.write(f"Test R2: {r2:.4f}\n")
        print(f"Test metrics saved to {metrics_path}")

        return y_pred_aligned, y_actual_aligned


    def plot_results(self, y_actual, y_pred, title="Actual vs Predicted PV Power"):
        """
        Plot actual vs predicted values for the test set.
        """
        plt.figure(figsize=(15, 6))
        plt.plot(y_actual.index, y_actual, label='Actual Power [W]', alpha=0.7)
        plt.plot(y_pred.index, y_pred, label='Predicted Power [W]', alpha=0.7)
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("PV Power [W]")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.results_dir, f'actual_vs_predicted_{self.timestamp}.png')
        plt.savefig(plot_path)
        print(f"Actual vs Predicted plot saved to {plot_path}")

        # Keep plot open after saving
        # plt.show() # Moved plt.show() to the end of the run method


    def plot_actual_vs_predicted_entire_test_set(self, y_actual_aligned, y_pred_aligned):
        """
        Plots the actual vs predicted PV power for the entire test set.
        """
        print("Plotting actual vs predicted for the entire test set...")
        self.plot_results(y_actual_aligned, y_pred_aligned, title="Actual vs Predicted PV Power (Entire Test Set)")


    def plot_training_history(self, history):
        """
        Plot training and validation loss over epochs.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MAE)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.results_dir, f'training_history_{self.timestamp}.png')
        plt.savefig(plot_path)
        print(f"Training history plot saved to {plot_path}")

        # Keep plot open after saving
        # plt.show() # Moved plt.show() to the end of the run method


    def run(self, data_path, sequence_length=144, batch_size=32, n_optuna_trials=50, final_epochs=100):
        """
        Run the full forecasting process: load, preprocess, split, optimize, train, evaluate, plot.
        """
        self.sequence_length = sequence_length
        self.batch_size = batch_size # Use the batch_size passed to run
        self.epochs = final_epochs # Set final training epochs

        # 1. Load and preprocess data
        df = self.load_data(data_path)
        df = self.create_time_features(df)

        # 2. Prepare features and split/scale data
        feature_info = self.prepare_features(df)
        scaled_data = self.split_and_scale_data(df, feature_info)

        X_train_scaled = scaled_data['X_train']
        y_train_scaled = scaled_data['y_train']
        X_val_scaled = scaled_data['X_val']
        y_val_scaled = scaled_data['y_val']
        X_test_scaled = scaled_data['X_test']
        y_test_scaled = scaled_data['y_test']
        original_test_df = scaled_data['original_test_df']
        target_scaler = scaled_data['scalers']['target']


        # 3. Create sequences
        print(f"\nCreating sequences with sequence length: {self.sequence_length}")
        X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train_scaled, self.sequence_length)
        X_val_seq, y_val_seq = self.create_sequences(X_val_scaled, y_val_scaled, self.sequence_length)
        X_test_seq, y_test_seq_eval = self.create_sequences(X_test_scaled, y_test_scaled, self.sequence_length) # y_test_seq_eval for evaluation metrics

        print(f"Train sequences shape: {X_train_seq.shape}, Target shape: {y_train_seq.shape}")
        print(f"Validation sequences shape: {X_val_seq.shape}, Target shape: {y_val_seq.shape}")
        print(f"Test sequences shape: {X_test_seq.shape}, Target shape: {y_test_seq_eval.shape}")


        # 4. Hyperparameter Optimization (only if n_optuna_trials > 0)
        if n_optuna_trials > 0:
            best_config = self.optimize_hyperparameters(
                X_train_seq, y_train_seq, X_val_seq, y_val_seq, n_trials=n_optuna_trials
            )
        else:
             print("\nHyperparameter optimization skipped.")
             best_config = self.config # Use default config if optimization is skipped


        # 5. Train Final Model with Best Config
        final_model, history = self.train_final_model(
            X_train_seq, y_train_seq, X_val_seq, y_val_seq, best_config
        )

        # 6. Evaluate Model
        # Use y_test_seq_eval directly for evaluation as create_sequences already aligns it
        # The target for a sequence ending at index i is the value at index i + sequence_length
        # So, y_test_seq_eval already contains the correct target values aligned with X_test_seq
        y_pred_aligned, y_actual_aligned = self.evaluate_model(
            final_model, X_test_seq, y_test_seq_eval, original_test_df, target_scaler
        )


        # 7. Plot Results
        self.plot_training_history(history)
        self.plot_actual_vs_predicted_entire_test_set(y_actual_aligned, y_pred_aligned)

        # Display all plots at the end
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM Model for PV Power Forecasting")
    parser.add_argument(
        '--data_path',
        type=str,
        default='data/processed/station_data_10min.parquet', # Updated default path
        help='Path to the input data file (parquet format).'
    )
    parser.add_argument(
        '--sequence_length',
        type=int,
        default=144, # Default 24 hours (144 * 10 minutes)
        help='Number of time steps (10-minute intervals) to look back for sequences.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32, # Added batch size argument
        help='Batch size for training.'
    )
    parser.add_argument(
        '--optuna_trials',
        type=int,
        default=100, # Increased default Optuna trials
        help='Number of Optuna trials for hyperparameter optimization. Set to 0 to skip optimization.'
    )
    parser.add_argument(
        '--final_epochs',
        type=int,
        default=200, # Increased default epochs for final training
        help='Number of epochs for final model training after optimization.'
    )
    # Added explicit --optimize flag
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Explicitly enable hyperparameter optimization (alternative to setting --optuna_trials > 0).'
    )

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
    BATCH_SIZE = args.batch_size # Use the parsed batch size
    # Determine if optimization is enabled: either --optimize flag is set OR --optuna_trials > 0
    OPTIMIZE = args.optimize or args.optuna_trials > 0
    N_TRIALS = args.optuna_trials if OPTIMIZE else 0 # Use trials only if optimizing
    FINAL_EPOCHS = args.final_epochs
    DATA_PATH = args.data_path

    print(f"Running LSTM model with parameters:")
    print(f"- Data path: {DATA_PATH}")
    print(f"- Data Resolution: 10 minutes (as per script design)")
    print(f"- Sequence length: {SEQUENCE_LENGTH} steps ({SEQUENCE_LENGTH*10/60:.2f} hours lookback)")
    print(f"- Batch size: {BATCH_SIZE}") # Print batch size
    print(f"- Max epochs for final training: {FINAL_EPOCHS}")
    print(f"- Hyperparameter optimization: {'Enabled' if OPTIMIZE else 'Disabled'}")
    if OPTIMIZE:
        print(f"- Number of optimization trials: {N_TRIALS}")

    # Initialize and run forecaster
    forecaster = LSTMHighResForecaster(
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE, # Pass the parsed batch size
        epochs=FINAL_EPOCHS
    )

    # Run pipeline
    # Pass n_optuna_trials to the run method
    forecaster.run(
        data_path=DATA_PATH,
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE, # Pass batch size to run method
        n_optuna_trials=N_TRIALS, # Pass the number of trials
        final_epochs=FINAL_EPOCHS
    )

    # The plot_results and plot_actual_vs_predicted_entire_test_set methods
    # within the run method already call plt.show() at the very end.

