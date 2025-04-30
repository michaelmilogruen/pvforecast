#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LSTM Model for PV Power Forecasting with High Resolution (10-minute) Data

This script implements an LSTM (Long Short-Term Memory) neural network for forecasting
photovoltaic power output using 10-minute resolution data. It includes data loading,
preprocessing, feature engineering, model training, and evaluation.

Key features:
1. Data preprocessing and feature scaling
2. Feature engineering for 10-minute resolution (including combined hour-minute cyclical feature)
3. LSTM model architecture
4. Hyperparameter optimization using Bayesian optimization with Optuna
5. Model evaluation and visualization
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

class LSTMHighResForecaster:
    def __init__(self, sequence_length=144, batch_size=32, epochs=50):
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

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.optuna_results_dir, exist_ok=True)

        # Timestamp for model versioning
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Default model configuration (will be updated by hyperparameter optimization)
        # Note: These ranges might need tuning based on the 10-minute data characteristics
        self.config = {
            'lstm_units': [64, 32],  # Default 2 LSTM layers
            'dense_units': [32, 16],   # Default 2 dense layers
            'dropout_rates': [0.2, 0.15], # Adaptive dropout rates for LSTM layers
            'dense_dropout_rates': [0.1, 0.05], # Adaptive dropout rates for dense layers
            'learning_rate': 0.001,
            'bidirectional': False,
            'batch_norm': True
        }

    def load_data(self, data_path):
        """
        Load data for training.

        Args:
            data_path: Path to the parquet file containing the data (expected 10-minute resolution)

        Returns:
            Loaded dataframe
        """
        print(f"Loading data from {data_path}...")
        try:
            df = pd.read_parquet(data_path)
            print(f"Data shape: {df.shape}")
            print(f"Data range: {df.index.min()} to {df.index.max()}")

            # Check for missing values
            missing_values = df.isna().sum()
            if missing_values.sum() > 0:
                print("Missing values in dataset:")
                print(missing_values[missing_values > 0])

                # Fill missing values
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
        Combines hour and minute into a single time-of-day feature for daily cycles.

        Args:
            df: DataFrame with a DatetimeIndex

        Returns:
            DataFrame with added time features
        """
        print("Creating time-based and cyclical features...")

        # Cyclical features for the day of the year (seasonal variation)
        df['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.0)
        df['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.0)

        # Combine hour and minute into a single time-of-day feature (in hours)
        # Example: 3:30 becomes 3.5
        df['time_of_day_hours'] = df.index.hour + df.index.minute / 60.0

        # Apply sinusoidal encoding to the combined time-of-day feature (daily cycle)
        df['time_of_day_sin'] = np.sin(2 * np.pi * df['time_of_day_hours'] / 24.0)
        df['time_of_day_cos'] = np.cos(2 * np.pi * df['time_of_day_hours'] / 24.0)

        # isNight calculation (simplified - ideally use sun position)
        # Assuming 'GlobalRadiation [W m-2]' exists in the data
        if 'GlobalRadiation [W m-2]' in df.columns:
            df['isNight'] = (df['GlobalRadiation [W m-2]'] < 1.0).astype(int)
            print("'isNight' feature created based on Global Radiation.")
        else:
            # If GlobalRadiation is not available, a more complex sun position calculation would be needed
            print("Warning: 'GlobalRadiation [W m-2]' not found. Cannot create 'isNight' feature based on radiation.")
            # For robustness, add a placeholder column if expected later
            df['isNight'] = 0 # Add a default column if not found, might impact model if relied upon


        return df


    def prepare_features(self, df):
        """
        Define and group features for the model based on scaling needs.

        Args:
            df: DataFrame with all raw and engineered features

        Returns:
            Dictionary with feature information
        """
        # Define features to use - ensure these columns exist after loading and engineering
        features = [
            'GlobalRadiation [W m-2]',
            'Temperature [degree_Celsius]',
            'WindSpeed [m s-1]',
            'ClearSkyIndex',
            # Replaced hour_sin/cos and minute_sin/cos with combined time_of_day_sin/cos
            'time_of_day_sin',
            'time_of_day_cos',
            'day_sin', # Keep day features for seasonal variation
            'day_cos', # Keep day features for seasonal variation
            'isNight'
        ]

        # Verify all intended features exist in the dataframe
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            print(f"Error: Missing features after loading and engineering: {missing_features}")
            # Decide how to handle missing critical features - exiting might be best
            # For this script, we'll remove them, but be aware this might impact model performance
            print(f"Removing missing features from the list: {missing_features}")
            features = [f for f in features if f in df.columns]
            if not features:
                print("Error: No valid features remaining after checking. Exiting.")
                exit()


        # Group features by scaling method based on common practices and your original code's logic
        # - Global Radiation and ClearSkyIndex: Often highly skewed - MinMaxScaler
        # - Temperature: Can be varying - StandardScaler is often robust
        # - Wind Speed: Can have outliers - RobustScaler is useful
        # - Time features (sin/cos) and isNight: Already in a bounded range - No scaling needed

        minmax_features = ['GlobalRadiation [W m-2]', 'ClearSkyIndex']
        standard_features = ['Temperature [degree_Celsius]']
        robust_features = ['WindSpeed [m s-1]']
        # Updated list for no scaling features
        no_scaling_features = ['time_of_day_sin', 'time_of_day_cos', 'day_sin', 'day_cos', 'isNight']

        # Filter to only include features that actually exist in the dataframe (after checking above)
        minmax_features = [f for f in minmax_features if f in features]
        standard_features = [f for f in standard_features if f in features]
        robust_features = [f for f in robust_features if f in features]
        no_scaling_features = [f for f in no_scaling_features if f in features]

        # Reconstruct the 'all_features' list in a defined order for consistent scaling
        all_features_ordered = minmax_features + standard_features + robust_features + no_scaling_features
        # Ensure no duplicates and all features are included
        all_features_ordered = list(dict.fromkeys(all_features_ordered)) # Removes duplicates and preserves order
        # Final check that all original 'features' are in the ordered list
        if set(features) != set(all_features_ordered):
            print("Warning: Feature list mismatch during ordering.")
            # Use the ordered list for consistency going forward
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
            Dictionary with split and scaled data and fitted scalers
        """
        # Split data into train, validation, and test sets by time
        # Use 70% for training, 15% for validation, 15% for testing
        # Use iloc for position-based slicing after ensuring index is sorted (parquet usually handles this)
        total_size = len(df)
        train_size = int(total_size * 0.7)
        val_size = int(total_size * 0.15)
        test_size = total_size - train_size - val_size # Use remaining for test

        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size : train_size + val_size]
        test_df = df.iloc[train_size + val_size : ]

        print(f"Total data size: {total_size}")
        print(f"Train set size: {len(train_df)}")
        print(f"Validation set size: {len(val_df)}")
        print(f"Test set size: {len(test_df)}")

        # Initialize scalers for different feature groups
        minmax_scaler = MinMaxScaler() if feature_info['minmax_features'] else None
        standard_scaler = StandardScaler() if feature_info['standard_features'] else None
        robust_scaler = RobustScaler() if feature_info['robust_features'] else None

        # Initialize target scaler (MinMaxScaler is common for output)
        target_scaler = MinMaxScaler(feature_range=(0, 1))

        # Fit scalers on training data only to prevent data leakage
        if minmax_scaler:
            minmax_scaler.fit(train_df[feature_info['minmax_features']])
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
                # Ensure feature exists before trying to access it
                if feature in df_subset.columns:
                    scaled_data[feature] = df_subset[feature].values
                else:
                    print(f"Warning: Feature '{feature}' not found in subset for scaling.")


            # Create DataFrame with all scaled features, maintaining original index
            # Ensure columns are in the same order as feature_info['all_features']
            # Use reindex to ensure column order, filling missing if any (though should not happen here)
            scaled_df = pd.DataFrame(scaled_data, index=df_subset.index).reindex(columns=feature_info['all_features'])

            return scaled_df

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

        Args:
            input_shape: Shape of input data (time_steps, features)
            trial: Optuna trial object for hyperparameter optimization (optional)

        Returns:
            Compiled LSTM model
        """
        # If trial is provided, use it to suggest hyperparameters
        if trial:
            # Suggest hyperparameters using Optuna
            # Number of layers
            num_lstm_layers = trial.suggest_int('num_lstm_layers', 1, 4) # Increased potential layers for complex data
            num_dense_layers = trial.suggest_int('num_dense_layers', 1, 3)

            # LSTM units for each layer - use consistent parameter ranges for each position
            lstm_units = []
            # Adjusting suggested unit ranges for potentially longer sequences/more complex data
            if num_lstm_layers >= 1:
                lstm_units.append(trial.suggest_int('lstm_units_1', 64, 512, step=32))
            if num_lstm_layers >= 2:
                lstm_units.append(trial.suggest_int('lstm_units_2', 32, 256, step=16))
            if num_lstm_layers >= 3:
                lstm_units.append(trial.suggest_int('lstm_units_3', 16, 128, step=8))
            if num_lstm_layers >= 4:
                lstm_units.append(trial.suggest_int('lstm_units_4', 8, 64, step=4))


            # Dense units for each layer - use consistent parameter ranges for each position
            dense_units = []
            if num_dense_layers >= 1:
                dense_units.append(trial.suggest_int('dense_units_1', 32, 128, step=16))
            if num_dense_layers >= 2:
                dense_units.append(trial.suggest_int('dense_units_2', 16, 64, step=8))
            if num_dense_layers >= 3:
                dense_units.append(trial.suggest_int('dense_units_3', 8, 32, step=4))

            # Suggest different dropout rates for each layer
            lstm_dropout_rates = []
            for i in range(1, num_lstm_layers + 1):
                lstm_dropout_rates.append(trial.suggest_float(f'lstm_dropout_rate_{i}', 0.1, 0.5, step=0.05))

            dense_dropout_rates = []
            for i in range(1, num_dense_layers): # No dropout after the last dense layer
                dense_dropout_rates.append(trial.suggest_float(f'dense_dropout_rate_{i}', 0.05, 0.3, step=0.05))

            learning_rate = trial.suggest_float('learning_rate', 5e-5, 5e-3, log=True) # Adjusted learning rate range slightly
            bidirectional = trial.suggest_categorical('bidirectional', [True, False])
            batch_norm = trial.suggest_categorical('batch_norm', [True, False]) # Allow optimizing batch_norm


            # Update config with suggested hyperparameters
            self.config['lstm_units'] = lstm_units
            self.config['dense_units'] = dense_units
            self.config['dropout_rates'] = lstm_dropout_rates
            self.config['dense_dropout_rates'] = dense_dropout_rates
            self.config['learning_rate'] = learning_rate
            self.config['bidirectional'] = bidirectional
            self.config['batch_norm'] = batch_norm

        # --- Corrected Variable Assignment ---
        # Retrieve hyperparameters from self.config AFTER potential update by trial
        lstm_units = self.config['lstm_units']
        dense_units = self.config['dense_units']
        dropout_rates = self.config['dropout_rates']
        dense_dropout_rates = self.config['dense_dropout_rates']
        learning_rate = self.config['learning_rate'] # Redundant, but keeps parameters grouped
        bidirectional = self.config['bidirectional']
        batch_norm = self.config['batch_norm']
        num_lstm_layers = len(lstm_units) # Re-calculate based on potentially changed list size
        num_dense_layers = len(dense_units) # Re-calculate based on potentially changed list size
        # --- End of Corrected Variable Assignment ---


        # Build the Sequential model
        model = Sequential()

        # --- Corrected Code Block for Adding LSTM Layers ---
        # Add the first LSTM layer (requires input_shape)
        if num_lstm_layers > 0:
            # Determine if the first layer should return sequences
            return_sequences_first = (num_lstm_layers > 1)

            if bidirectional:
                # Add Bidirectional wrapper for the first layer, specifying input_shape in the wrapper
                # Pass the LSTM layer instance to the Bidirectional wrapper constructor
                model.add(Bidirectional(LSTM(units=lstm_units[0], return_sequences=return_sequences_first),
                                        input_shape=input_shape))
            else:
                # Add the first LSTM layer directly, specifying input_shape in its constructor
                model.add(LSTM(units=lstm_units[0],
                               return_sequences=return_sequences_first,
                               input_shape=input_shape)) # Pass input_shape here


            # Add Batch Normalization and Dropout after the first layer if configured
            if batch_norm:
                model.add(BatchNormalization())

            # Apply dropout after the first LSTM/Bidirectional layer
            # Use dropout rate corresponding to index 0 if available, otherwise 0.0
            dropout_rate_first = dropout_rates[0] if dropout_rates and len(dropout_rates) > 0 else 0.0
            if dropout_rate_first > 1e-6: # Add dropout if rate is non-negligible
                model.add(Dropout(dropout_rate_first))


            # Add subsequent LSTM layers (if any, they infer input shape)
            for i in range(1, num_lstm_layers):
                is_last_lstm = (i == num_lstm_layers - 1)
                # Create subsequent LSTM layer instance
                subsequent_lstm_layer_instance = LSTM(units=lstm_units[i], return_sequences=not is_last_lstm)

                if bidirectional:
                    model.add(Bidirectional(subsequent_lstm_layer_instance))
                else:
                    model.add(subsequent_lstm_layer_instance)

                # Add Batch Normalization and Dropout after subsequent layers
                if batch_norm:
                    model.add(BatchNormalization())

                # Apply dropout after subsequent LSTM/Bidirectional layers
                # Use dropout rate corresponding to this layer index, or the last rate if index is out of bounds
                dropout_rate_subsequent = dropout_rates[i] if i < len(dropout_rates) else (dropout_rates[-1] if dropout_rates else 0.0)
                if dropout_rate_subsequent > 1e-6: # Add dropout if rate is non-negligible
                    model.add(Dropout(dropout_rate_subsequent))

        # --- End of Corrected Code Block for Adding LSTM Layers ---


        # Add Dense layers
        for i in range(num_dense_layers):
            model.add(Dense(dense_units[i], activation='relu'))

            if batch_norm:
                model.add(BatchNormalization())

            if i < num_dense_layers - 1: # No dropout after the last dense layer
                # Use dense dropout rate corresponding to this layer index, or the last rate
                dense_dropout_rate_current = dense_dropout_rates[i] if i < len(dense_dropout_rates) else (dense_dropout_rates[-1] if dense_dropout_rates else 0.0)
                if dense_dropout_rate_current > 1e-6: # Add dropout if rate is non-negligible
                    model.add(Dropout(dense_dropout_rate_current))


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
            patience=15, # Increased patience slightly for potentially longer training
            restore_best_weights=True
        )
        callbacks.append(early_stopping)

        # Model checkpoint (only for final model, not during hyperparameter search)
        if trial is None:
            model_checkpoint = ModelCheckpoint(
                filepath=os.path.join(self.model_dir, f'best_model_{self.timestamp}.keras'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
            callbacks.append(model_checkpoint)

        # Learning rate reduction
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7, # Adjusted patience
            min_lr=1e-7, # Adjusted min_lr
            verbose=0 if trial else 1 # Less verbose during hyperparameter search
        )
        callbacks.append(reduce_lr)

        # Add Optuna pruning callback if trial is provided
        if trial:
            # Pruning is based on the validation loss
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
        # Clear previous TensorFlow session to free up memory
        tf.keras.backend.clear_session()

        # Build model with hyperparameters from trial
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        model = self.build_model(input_shape, trial)

        # Create callbacks with pruning
        callbacks = self.create_callbacks(trial)

        # Train model with fewer epochs for hyperparameter search (adjust as needed based on time/resources)
        # With 10-minute data, 20 epochs might still take a while. Consider reducing if needed.
        epochs_for_trial = 15 # Reduced epochs for faster trials

        try:
            history = model.fit(
                X_train_seq, y_train_seq,
                epochs=epochs_for_trial,
                batch_size=self.batch_size, # Use the class's batch size
                validation_data=(X_val_seq, y_val_seq),
                callbacks=callbacks,
                verbose=0 # Silent training during hyperparameter search
            )

            # Return the best validation loss reached during this trial
            # Use min(history.history['val_loss']) in case it improved then got worse
            validation_loss = min(history.history['val_loss'])

        except Exception as e:
            print(f"Trial {trial.number} failed due to error: {e}")
            # Prune the trial if it fails
            raise optuna.exceptions.TrialPruned()

        return validation_loss

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
        print(f"\nStarting hyperparameter optimization with {n_trials} trials using Optuna...")

        # Create a study object with pruning
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5, # Allow first few trials to run without pruning
            n_warmup_steps=5,   # Number of steps before pruning can start in a trial
            interval_steps=1    # Check for pruning every epoch
        )

        # Use a SQLite database to store study results, allowing continuation if interrupted
        db_path = os.path.join(self.optuna_results_dir, f'optuna_study_{self.timestamp}.db')
        study = optuna.create_study(
            direction='minimize',
            pruner=pruner,
            study_name=f'lstm_10min_study_{self.timestamp}',
            storage=f'sqlite:///{db_path}', # Save study to database
            load_if_exists=True # Load existing study if it exists
        )

        print(f"Optuna study stored at: {db_path}")
        if len(study.trials) > 0:
            print(f"Loaded existing study with {len(study.trials)} trials.")

        # Define the objective function
        objective_func = lambda trial: self.objective(
            trial, X_train_seq, y_train_seq, X_val_seq, y_val_seq
        )

        # Run the optimization, considering already completed trials if loading from database
        remaining_trials = n_trials - len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        if remaining_trials > 0:
            print(f"Running {remaining_trials} new optimization trials...")
            study.optimize(objective_func, n_trials=remaining_trials, show_progress_bar=True)
        else:
            print("Optimization already completed for the specified number of trials.")


        # Get best parameters
        if study.best_trial is None:
            print("No trials completed successfully.")
            return self.config # Return default config if no best trial
        else:
            best_params = study.best_params
            print("\nBest hyperparameters found:")
            for param, value in best_params.items():
                print(f"{param}: {value}")

            # Update config with best parameters from the study
            # Need to extract list parameters based on the number of layers found
            num_lstm_layers = best_params.get('num_lstm_layers', len(self.config['lstm_units'])) # Use default if not in best_params (e.g. if load_if_exists)
            num_dense_layers = best_params.get('num_dense_layers', len(self.config['dense_units']))

            lstm_units = [best_params[f'lstm_units_{i+1}'] for i in range(num_lstm_layers) if f'lstm_units_{i+1}' in best_params]
            dense_units = [best_params[f'dense_units_{i+1}'] for i in range(num_dense_layers) if f'dense_units_{i+1}' in best_params]
            lstm_dropout_rates = [best_params[f'lstm_dropout_rate_{i+1}'] for i in range(num_lstm_layers) if f'lstm_dropout_rate_{i+1}' in best_params]
            dense_dropout_rates = [best_params[f'dense_dropout_rate_{i+1}'] for i in range(num_dense_layers - 1) if f'dense_dropout_rate_{i+1}' in best_params] # Note: dense_dropout_rate is for intermediate layers

            self.config['lstm_units'] = lstm_units if lstm_units else self.config['lstm_units']
            self.config['dense_units'] = dense_units if dense_units else self.config['dense_units']
            self.config['dropout_rates'] = lstm_dropout_rates if lstm_dropout_rates else self.config['dropout_rates']
            self.config['dense_dropout_rates'] = dense_dropout_rates if dense_dropout_rates else self.config['dense_dropout_rates']
            self.config['learning_rate'] = best_params.get('learning_rate', self.config['learning_rate'])
            self.config['bidirectional'] = best_params.get('bidirectional', self.config['bidirectional'])
            self.config['batch_norm'] = best_params.get('batch_norm', self.config['batch_norm'])


            # Save hyperparameter search results plots
            try:
                self.plot_optuna_results(study)
            except Exception as e:
                print(f"Warning: Could not generate Optuna plots. Error: {e}")


            # Save best parameters to file
            best_params_path = os.path.join(self.model_dir, f'best_params_{self.timestamp}.txt')
            with open(best_params_path, 'w', encoding='utf-8') as f:
                f.write("Optimized Hyperparameters:\n")
                for param, value in self.config.items(): # Save the config dict after updating it
                    f.write(f"{param}: {value}\n")
            print(f"Best hyperparameters saved to {best_params_path}")

            return self.config # Return the updated config

    def plot_optuna_results(self, study):
        """
        Plot and save Optuna optimization results.

        Args:
            study: Optuna study object
        """
        print(f"Saving Optuna visualization plots to {self.optuna_results_dir}/")

        # Plot optimization history
        try:
            fig1 = optuna.visualization.plot_optimization_history(study)
            fig1.write_image(os.path.join(self.optuna_results_dir, f'optimization_history_{self.timestamp}.png'))
            # fig1.show() # Uncomment to show interactively if needed
        except Exception as e:
            print(f"Warning: Could not plot optimization history. Error: {e}")

        # Plot parameter importances
        try:
            fig2 = optuna.visualization.plot_param_importances(study)
            fig2.write_image(os.path.join(self.optuna_results_dir, f'param_importances_{self.timestamp}.png'))
            # fig2.show()
        except Exception as e:
            print(f"Warning: Could not plot parameter importances. Error: {e}")

        # Plot parallel coordinate plot (can be slow/complex for many trials/params)
        try:
            fig3 = optuna.visualization.plot_parallel_coordinate(study)
            fig3.write_image(os.path.join(self.optuna_results_dir, f'parallel_coordinate_{self.timestamp}.png'))
            # fig3.show()
        except Exception as e:
            print(f"Warning: Could not plot parallel coordinate. Error: {e}")

        # Plot slice plots for key hyperparameters that exist in the study
        # Get all parameter names from the study
        study_params = list(study.best_params.keys())
        plottable_params = [p for p in study_params if 'units' in p or 'rate' in p or 'learning_rate' in p or p in ['num_lstm_layers', 'num_dense_layers', 'bidirectional', 'batch_norm']]

        for param in plottable_params:
            try:
                fig = optuna.visualization.plot_slice(study, params=[param])
                fig.write_image(os.path.join(self.optuna_results_dir, f'slice_{param}_{self.timestamp}.png'))
                # fig.show()
            except Exception as e:
                print(f"Warning: Could not plot slice for {param}. Error: {e}")


    def evaluate_model(self, model, X_test_seq, y_test, target_scaler=None):
        """
        Evaluate the model and return metrics.

        Args:
            model: Trained LSTM model
            X_test_seq: Test sequences
            y_test: True target values (scaled)
            target_scaler: Scaler for the target variable (optional, to get original scale metrics)

        Returns:
            Dictionary of evaluation metrics and actual/predicted values
        """
        print("\nMaking predictions on the test set...")
        y_pred = model.predict(X_test_seq)

        # Ensure y_test and y_pred have the same shape (e.g., (n_samples, 1))
        y_test = y_test.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

        # Calculate metrics on scaled data
        mse_scaled = mean_squared_error(y_test, y_pred)
        rmse_scaled = np.sqrt(mse_scaled)
        mae_scaled = mean_absolute_error(y_test, y_pred)
        r2_scaled = r2_score(y_test, y_pred)

        # Calculate MAPE (Mean Absolute Percentage Error) for scaled data
        # Avoid division by zero or very small numbers
        epsilon = 1e-8 # Small constant to avoid division by zero
        mape_scaled = np.mean(np.abs((y_test - y_pred) / (y_test + epsilon))) * 100

        # Calculate SMAPE (Symmetric Mean Absolute Percentage Error) for scaled data
        # This metric is symmetric and handles zero values better
        smape_scaled = np.mean(2.0 * np.abs(y_pred - y_test) / (np.abs(y_pred) + np.abs(y_test) + epsilon)) * 100

        print("\nModel Evaluation (Scaled):")
        print(f"Mean Squared Error (MSE): {mse_scaled:.6f}")
        print(f"Root Mean Squared Error (RMSE): {rmse_scaled:.6f}")
        print(f"Mean Absolute Error (MAE): {mae_scaled:.6f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape_scaled:.2f}%")
        print(f"Symmetric Mean Absolute Percentage Error (SMAPE): {smape_scaled:.2f}%")
        print(f"R² Score: {r2_scaled:.4f}")


        results = {
            'mse_scaled': mse_scaled,
            'rmse_scaled': rmse_scaled,
            'mae_scaled': mae_scaled,
            'mape_scaled': mape_scaled,
            'smape_scaled': smape_scaled,
            'r2_scaled': r2_scaled,
            'y_test_scaled': y_test,
            'y_pred_scaled': y_pred
        }

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
            # Avoid division by zero or very small numbers
            epsilon = 1e-8 # Small constant to avoid division by zero
            mape = np.mean(np.abs((y_test_inv - y_pred_inv) / (y_test_inv + epsilon))) * 100
            smape = np.mean(2.0 * np.abs(y_pred_inv - y_test_inv) / (np.abs(y_pred_inv) + np.abs(y_pred_inv) + epsilon)) * 100


            print("\nModel Evaluation (Original Scale):")
            print(f"Mean Squared Error (MSE): {mse:.2f}")
            print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
            print(f"Mean Absolute Error (MAE): {mae:.2f}")
            print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
            print(f"Symmetric Mean Absolute Percentage Error (SMAPE): {smape:.2f}%")
            print(f"R² Score: {r2:.4f}")

            results.update({
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'smape': smape,
                'r2': r2,
                'y_test_inv': y_test_inv,
                'y_pred_inv': y_pred_inv
            })

        return results


    def plot_results(self, history, evaluation_results):
        """
        Plot and save training history and prediction results.

        Args:
            history: Training history object
            evaluation_results: Dictionary from model evaluation
        """
        print(f"Saving training history and prediction plots to {self.results_dir}/")

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
        plt.close() # Close plot to free memory


        # Check if we have inverse-transformed data for original scale plots
        has_inverse_transform = 'y_test_inv' in evaluation_results

        # Plot predictions vs actual (scaled) - Sample a smaller portion for clarity if needed
        plt.figure(figsize=(14, 7))

        sample_size = min(1000, len(evaluation_results['y_test_scaled'])) # Sample size for plotting
        indices = np.arange(sample_size)

        plt.plot(indices, evaluation_results['y_test_scaled'][:sample_size], 'b-', label='Actual (Scaled)')
        plt.plot(indices, evaluation_results['y_pred_scaled'][:sample_size], 'r-', label='Predicted (Scaled)')
        plt.title(f'Actual vs Predicted PV Power (Scaled) - Sample ({sample_size} points)')
        plt.xlabel(f'Time Steps ({self.sequence_length}-step lookback)')
        plt.ylabel('Power Output (Scaled)')
        plt.legend()
        plt.grid(True)

        plt.savefig(os.path.join(self.results_dir, f'lstm_10min_predictions_scaled_{self.timestamp}.png'))
        plt.close() # Close plot


        # If we have inverse-transformed data, plot on original scale
        if has_inverse_transform:
            # Plot predictions vs actual (original scale)
            plt.figure(figsize=(14, 7))

            # Use the same sample size and indices as scaled plot
            plt.plot(indices, evaluation_results['y_test_inv'][:sample_size], 'b-', label='Actual Power Output')
            plt.plot(indices, evaluation_results['y_pred_inv'][:sample_size], 'r-', label='Predicted Power Output')
            plt.title(f'Actual vs Predicted PV Power (W) - Sample ({sample_size} points)')
            plt.xlabel(f'Time Steps ({self.sequence_length}-step lookback)')
            plt.ylabel('Power Output (W)')
            plt.legend()
            plt.grid(True)

            plt.savefig(os.path.join(self.results_dir, f'lstm_10min_predictions_{self.timestamp}.png'))
            plt.close() # Close plot

            # Create scatter plot (original scale)
            plt.figure(figsize=(10, 8))
            # Use a smaller sample size for scatter plot if the test set is very large
            scatter_sample_size = min(5000, len(evaluation_results['y_test_inv']))
            scatter_indices = np.random.choice(len(evaluation_results['y_test_inv']), scatter_sample_size, replace=False) # Sample randomly

            plt.scatter(evaluation_results['y_test_inv'][scatter_indices], evaluation_results['y_pred_inv'][scatter_indices], alpha=0.5, s=5) # Reduced marker size
            # Plot a diagonal line for reference
            max_val = max(evaluation_results['y_test_inv'].max(), evaluation_results['y_pred_inv'].max())
            min_val = min(evaluation_results['y_test_inv'].min(), evaluation_results['y_test_inv'].min())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

            plt.title(f'Actual vs Predicted PV Power (W) - Scatter Plot ({scatter_sample_size} points sampled)')
            plt.xlabel('Actual Power Output (W)')
            plt.ylabel('Predicted Power Output (W)')
            plt.grid(True)
            plt.axis('equal') # Ensure aspect ratio is equal
            plt.gca().set_aspect('equal', adjustable='box') # Alternative way to ensure equal aspect ratio

            plt.savefig(os.path.join(self.results_dir, f'lstm_10min_scatter_{self.timestamp}.png'))
            plt.close() # Close plot
        else:
            # Create scatter plot (scaled) if original scale data is not available
            plt.figure(figsize=(10, 8))
            scatter_sample_size = min(5000, len(evaluation_results['y_test_scaled']))
            scatter_indices = np.random.choice(len(evaluation_results['y_test_scaled']), scatter_sample_size, replace=False)

            plt.scatter(evaluation_results['y_test_scaled'][scatter_indices], evaluation_results['y_pred_scaled'][scatter_indices], alpha=0.5, s=5)
            max_val = max(evaluation_results['y_test_scaled'].max(), evaluation_results['y_pred_scaled'].max())
            min_val = min(evaluation_results['y_test_scaled'].min(), evaluation_results['y_test_scaled'].min())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

            plt.title(f'Actual vs Predicted (Scaled) - Scatter Plot ({scatter_sample_size} points sampled)')
            plt.xlabel('Actual Power Output (Scaled)')
            plt.ylabel('Predicted Power Output (Scaled)')
            plt.grid(True)
            plt.axis('equal')
            plt.gca().set_aspect('equal', adjustable='box')

            plt.savefig(os.path.join(self.results_dir, f'lstm_10min_scatter_scaled_{self.timestamp}.png'))
            plt.close() # Close plot

        print("Plots saved.")


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

        # Load data
        df = self.load_data(data_path)

        # Create time features (assuming they are not already in the raw data)
        # Ensure this is done BEFORE prepare_features
        df = self.create_time_features(df)


        # Prepare features
        feature_info = self.prepare_features(df)

        # Split and scale data
        data = self.split_and_scale_data(df, feature_info)

        # Clear original dataframe to free memory after splitting/scaling
        del df


        # Create sequences
        # Note: The sequence_length is defined in __init__ (default 144 for 24 hours)
        print(f"\nCreating sequences with sequence length: {self.sequence_length} ({self.sequence_length*10/60:.2f} hours lookback)")

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
            # Optimization will update self.config with the best parameters found
            self.optimize_hyperparameters(
                X_train_seq, y_train_seq, X_val_seq, y_val_seq, n_trials=n_trials
            )
            print(f"\nOptimized hyperparameters loaded into configuration.")

        else:
            print("\nHyperparameter optimization skipped. Using default or previously loaded configuration.")
            # If not optimizing, ensure a config exists (e.g., load from file or use default)
            # For this script, default config is already initialized


        # Build final model using the current config (either default or optimized)
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
            epochs=self.epochs, # Use the full number of epochs
            batch_size=self.batch_size,
            validation_data=(X_val_seq, y_val_seq),
            callbacks=callbacks,
            verbose=1
        )

        # Load the best model saved by the checkpoint callback
        best_model_path = os.path.join(self.model_dir, f'best_model_{self.timestamp}.keras')
        if os.path.exists(best_model_path):
            print(f"\nLoading best model from {best_model_path}")
            model = tf.keras.models.load_model(best_model_path)
        else:
            print("\nWarning: Best model checkpoint not found. Using the model from the last training epoch.")


        # Evaluate model
        print("\nEvaluating final model...")
        evaluation_results = self.evaluate_model(
            model, X_test_seq, y_test_seq, data['scalers']['target'] # Use y_test_seq for evaluation target
        )

        # Plot results
        self.plot_results(history, evaluation_results)

        # Save model summary and details to file
        self.save_model_summary(model, feature_info, evaluation_results, data_path)


        return evaluation_results

    def save_model_summary(self, model, feature_info, evaluation_results, data_path):
        """
        Saves the model architecture, data info, scaling, and evaluation results to a text file.
        """
        summary_path = os.path.join(self.model_dir, f'model_summary_{self.timestamp}.txt')
        print(f"\nSaving model summary and details to {summary_path}")

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"# LSTM High Resolution Model ({self.sequence_length}-step lookback)\n\n")
            f.write(f"Timestamp: {self.timestamp}\n\n")

            f.write("## Model Architecture\n\n")
            model.summary(print_fn=lambda x: f.write(x + '\n'))

            f.write("\n\n## Data Information\n\n")
            f.write(f"- Data Source: {data_path}\n")
            f.write(f"- Data Resolution: 10 minutes\n")
            f.write(f"- Sequence Length: {self.sequence_length} steps ({self.sequence_length * 10 / 60:.2f} hours lookback)\n")
            f.write(f"- Training samples (sequences): {evaluation_results['y_test_scaled'].shape[0] if 'y_test_scaled' in evaluation_results else 'N/A'} (based on test set size after sequence creation)\n") # Approximate count based on test set length after sequence creation


            f.write("\n\n## Features Used\n")
            f.write("The model was trained on the following features:\n")
            for feature in feature_info['all_features']:
                f.write(f"- {feature}\n")
            f.write(f"\nTarget variable: {feature_info['target']}\n\n")


            f.write("## Scaler Structure and Implementation\n\n")
            f.write("Multiple scalers were used for different feature types:\n\n")

            if feature_info['minmax_features']:
                f.write("### MinMaxScaler\n")
                f.write("Applied to: " + ", ".join(feature_info['minmax_features']) + "\n\n")

            if feature_info['standard_features']:
                f.write("### StandardScaler\n")
                f.write("Applied to: " + ", ".join(feature_info['standard_features']) + "\n\n")

            if feature_info['robust_features']:
                f.write("### RobustScaler\n")
                f.write("Applied to: " + ", ".join(feature_info['robust_features']) + "\n\n")

            if feature_info['no_scaling_features']:
                f.write("### No Scaling\n")
                f.write("Applied to: " + ", ".join(feature_info['no_scaling_features']) + "\n\n")

            f.write("### Target Scaler\n")
            f.write(f"A MinMaxScaler was used for the target variable ({feature_info['target']}).\n")
            f.write("Scalers were fitted only on the training data and saved.\n\n")

            f.write("## Hyperparameters\n\n")
            f.write("Configuration used for the final model:\n")
            for param, value in self.config.items():
                f.write(f"- {param}: {value}\n")
            f.write("\n")
            if hasattr(self, 'best_params'): # Include if optimization was run
                f.write("Note: These hyperparameters were determined by Optuna Bayesian Optimization.\n\n")


            f.write("## Evaluation Metrics (on Test Set)\n\n")
            if 'rmse' in evaluation_results: # Check if original scale metrics are available
                f.write("### Original Scale Metrics\n")
                f.write(f"- RMSE: {evaluation_results['rmse']:.2f}\n")
                f.write(f"- MAE: {evaluation_results['mae']:.2f}\n")
                f.write(f"- MAPE: {evaluation_results['mape']:.2f}%\n")
                f.write(f"- SMAPE: {evaluation_results['smape']:.2f}%\n")
                f.write(f"- R² Score: {evaluation_results['r2']:.4f}\n\n")
            else: # Fallback to scaled metrics if original scale not available
                f.write("### Scaled Metrics\n")
                f.write(f"- MSE: {evaluation_results['mse']:.6f}\n")
                f.write(f"- RMSE: {evaluation_results['rmse']:.6f}\n")
                f.write(f"- MAE: {evaluation_results['mae']:.6f}\n")
                f.write(f"- MAPE: {evaluation_results['mape']:.2f}%\n")
                f.write(f"- SMAPE: {evaluation_results['smape']:.2f}%\n")
                f.write(f"- R² Score: {evaluation_results['r2']:.4f}\n\n")


            f.write("## Inference Data Preparation for this Model\n\n")
            f.write("To prepare 10-minute data for inference using this trained model, you must follow these steps precisely:\n\n")
            f.write("1.  Load your raw 10-minute data, ensuring it has the same meteorological features used during training.\n")
            f.write("2.  Calculate the following time-based and cyclical features:\n")
            f.write("    -   `time_of_day_hours = hour + minute / 60.0`\n")
            f.write("    -   `time_of_day_sin = sin(2 * pi * time_of_day_hours / 24.0)`\n")
            f.write("    -   `time_of_day_cos = cos(2 * pi * time_of_day_hours / 24.0)`\n")
            f.write("    -   `day_sin = sin(2 * pi * day_of_year / 365.0)`\n")
            f.write("    -   `day_cos = cos(2 * pi * day_of_year / 365.0)`\n")
            f.write(f"    -   `isNight` (e.g., based on Global Radiation < 1.0 W/m² or a sun position calculation).\n")
            f.write("3.  Ensure the 'ClearSkyIndex' feature is present or calculated for the 10-minute data.\n")
            f.write("4.  Order the features exactly as they were ordered during training:\n")
            f.write("    " + ", ".join(feature_info['all_features']) + "\n")
            f.write("5.  Load the saved scalers (`minmax_scaler_*.pkl`, `standard_scaler_*.pkl`, `robust_scaler_*.pkl`) from the model directory.\n")
            f.write("6.  Apply the corresponding loaded scalers to the appropriate feature columns as done during training.\n")
            f.write("7.  Create input sequences of length **{self.sequence_length}** from the scaled feature data.\n")
            f.write("8.  Feed these sequences to the loaded model for prediction.\n")
            f.write("9.  Load the saved target scaler (`target_scaler_*.pkl`).\n")
            f.write("10. Use the target scaler to **inverse transform** the model's predictions back to the original power (Watts) scale.\n\n")
            f.write("Following these steps is crucial for the model to produce meaningful predictions on new data.")


# --- Main Execution Block ---
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run LSTM model for 10-minute PV forecasting with Bayesian hyperparameter optimization')
    parser.add_argument('--optimize', action='store_true', help='Perform hyperparameter optimization')
    # Updated default sequence length for 10-minute data (24 hours)
    parser.add_argument('--sequence_length', type=int, default=144, help='Sequence length (10-minute intervals to look back, default 144 for 24 hours)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (adjust based on GPU memory)')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of epochs for final model training') # Increased default epochs
    parser.add_argument('--trials', type=int, default=50, help='Number of Optuna optimization trials')
    parser.add_argument('--data_path', type=str, default='data/station_data_10min.parquet', help='Path to the 10-minute resolution data file')

    args = parser.parse_args()

    # Check for GPU availability
    print("TensorFlow version:", tf.__version__)
    gpu_devices = tf.config.list_physical_devices('GPU')
    print("GPU Available:", "Yes" if gpu_devices else "No")
    if not gpu_devices:
        print("Warning: No GPU devices found. Training may be very slow on CPU.")


    # Set memory growth to avoid memory allocation issues on GPU
    if gpu_devices:
        try:
            for gpu in gpu_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory growth set to True for all GPUs")
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")

    # Set parameters from arguments
    SEQUENCE_LENGTH = args.sequence_length
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    OPTIMIZE = args.optimize
    N_TRIALS = args.trials
    DATA_PATH = args.data_path


    print(f"\nRunning LSTM model with parameters:")
    print(f"- Data path: {DATA_PATH}")
    print(f"- Data Resolution: 10 minutes")
    print(f"- Sequence length: {SEQUENCE_LENGTH} steps ({SEQUENCE_LENGTH*10/60:.2f} hours lookback)")
    print(f"- Batch size: {BATCH_SIZE}")
    print(f"- Max epochs (final training): {EPOCHS}")
    print(f"- Hyperparameter optimization: {'Enabled' if OPTIMIZE else 'Disabled'}")
    if OPTIMIZE:
        print(f"- Number of optimization trials: {N_TRIALS}")

    # Initialize and run forecaster
    forecaster = LSTMHighResForecaster(
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS # Max epochs for final training
    )

    # Run pipeline
    metrics = forecaster.run_pipeline(
        data_path=DATA_PATH,
        optimize_hyperparams=OPTIMIZE,
        n_trials=N_TRIALS
    )

    # Print final results (will be from original scale if target scaler was used)
    print("\n--- Final Evaluation Results (Test Set) ---")
    if 'rmse' in metrics:
        print(f"RMSE: {metrics['rmse']:.2f}")
        print(f"MAE: {metrics['mae']:.2f}")
        print(f"MAPE: {metrics['mape']:.2f}%")
        print(f"SMAPE: {metrics['smape']:.2f}%")
        print(f"R²: {metrics['r2']:.4f}")
    else: # Fallback to scaled metrics if original scale not available
        print(f"RMSE (Scaled): {metrics['rmse_scaled']:.6f}")
        print(f"MAE (Scaled): {metrics['mae_scaled']:.6f}")
        print(f"MAPE (Scaled): {metrics['mape_scaled']:.2f}%")
        print(f"SMAPE (Scaled): {metrics['smape_scaled']:.2f}%")
        print(f"R² (Scaled): {metrics['r2_scaled']:.4f}")

    print("-" * 35)
    print(f"Model artifacts saved to {forecaster.model_dir}/")
    print(f"Results plots saved to {forecaster.results_dir}/")
    if OPTIMIZE:
        print(f"Optuna results saved to {forecaster.optuna_results_dir}/")