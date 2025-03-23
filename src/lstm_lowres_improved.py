#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Improved LSTM Model for PV Power Forecasting with Low Resolution (1-hour) Data

This script implements an enhanced LSTM neural network for forecasting
photovoltaic power output using 1-hour resolution data. Key improvements:
1. Target scaling to handle the skewed power distribution
2. More complex model architecture
3. Longer sequence length
4. Additional regularization techniques
5. Hyperparameter optimization using successive halving
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
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV
from scikeras.wrappers import KerasRegressor
from scipy.stats import randint, uniform
import joblib
import os
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class LSTMLowResImproved:
    def __init__(self, sequence_length=48, batch_size=32, epochs=100):
        """
        Initialize the improved LSTM forecaster for low-resolution data.
        
        Args:
            sequence_length: Number of time steps to look back (default: 48 hours - 2 days)
            batch_size: Batch size for training
            epochs: Maximum number of epochs for training
        """
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Create directories for model and results
        os.makedirs('models/lstm_lowres_improved', exist_ok=True)
        os.makedirs('results/lstm_lowres_improved', exist_ok=True)
        
        # Timestamp for model versioning
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Enhanced model configuration
        self.config = {
            'lstm_units': [128, 64, 32],
            'dense_units': [32, 16],
            'dropout_rates': [0.3, 0.3, 0.3],
            'learning_rate': 0.001,
            'bidirectional': True,
            'batch_norm': True
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
        Split data into train, validation, and test sets, then scale features and target.
        
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
            joblib.dump(minmax_scaler, f'models/lstm_lowres_improved/minmax_scaler_{self.timestamp}.pkl')
        
        if standard_scaler:
            standard_scaler.fit(train_df[feature_info['standard_features']])
            joblib.dump(standard_scaler, f'models/lstm_lowres_improved/standard_scaler_{self.timestamp}.pkl')
        
        if robust_scaler:
            robust_scaler.fit(train_df[feature_info['robust_features']])
            joblib.dump(robust_scaler, f'models/lstm_lowres_improved/robust_scaler_{self.timestamp}.pkl')
        
        # Fit target scaler on training data
        target_scaler.fit(train_df[[feature_info['target']]])
        joblib.dump(target_scaler, f'models/lstm_lowres_improved/target_scaler_{self.timestamp}.pkl')
        
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
    
    def create_model_builder(self, input_shape):
        """
        Create a model builder function for hyperparameter optimization.
        
        Args:
            input_shape: Shape of input data (time_steps, features)
            
        Returns:
            Function that builds and returns a compiled model
        """
        def build_model_for_optimization(
            lstm_units_1=128,
            lstm_units_2=64,
            lstm_units_3=32,
            dense_units_1=32,
            dense_units_2=16,
            dropout_rate=0.3,
            learning_rate=0.001,
            bidirectional=True,
            batch_norm=True
        ):
            model = Sequential()
            
            # First LSTM layer
            if bidirectional:
                model.add(Bidirectional(
                    LSTM(units=lstm_units_1, return_sequences=True),
                    input_shape=input_shape
                ))
            else:
                model.add(LSTM(
                    units=lstm_units_1,
                    return_sequences=True,
                    input_shape=input_shape
                ))
            
            if batch_norm:
                model.add(BatchNormalization())
            
            model.add(Dropout(dropout_rate))
            
            # Second LSTM layer
            if bidirectional:
                model.add(Bidirectional(
                    LSTM(units=lstm_units_2, return_sequences=True)
                ))
            else:
                model.add(LSTM(
                    units=lstm_units_2,
                    return_sequences=True
                ))
            
            if batch_norm:
                model.add(BatchNormalization())
            
            model.add(Dropout(dropout_rate))
            
            # Last LSTM layer
            if bidirectional:
                model.add(Bidirectional(
                    LSTM(units=lstm_units_3, return_sequences=False)
                ))
            else:
                model.add(LSTM(
                    units=lstm_units_3,
                    return_sequences=False
                ))
            
            if batch_norm:
                model.add(BatchNormalization())
            
            model.add(Dropout(dropout_rate))
            
            # Dense layers
            model.add(Dense(dense_units_1, activation='relu'))
            if batch_norm:
                model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
            
            model.add(Dense(dense_units_2, activation='relu'))
            if batch_norm:
                model.add(BatchNormalization())
            
            # Output layer
            model.add(Dense(1))
            
            # Compile model
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            
            return model
        
        return build_model_for_optimization
    
    def optimize_hyperparameters(self, X_train_seq, y_train_seq, X_val_seq, y_val_seq):
        """
        Optimize hyperparameters using successive halving.
        
        Args:
            X_train_seq: Training sequences
            y_train_seq: Training targets
            X_val_seq: Validation sequences
            y_val_seq: Validation targets
            
        Returns:
            Dictionary with optimized hyperparameters
        """
        print("\nStarting hyperparameter optimization using successive halving...")
        
        # Create model builder
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        model_builder = self.create_model_builder(input_shape)
        
        # Create KerasRegressor
        model = KerasRegressor(
            build_fn=model_builder,
            epochs=20,  # Use fewer epochs for hyperparameter search
            batch_size=self.batch_size,
            verbose=0
        )
        
        # Define parameter distributions
        param_distributions = {
            'lstm_units_1': randint(32, 256),
            'lstm_units_2': randint(16, 128),
            'lstm_units_3': randint(8, 64),
            'dense_units_1': randint(16, 64),
            'dense_units_2': randint(8, 32),
            'dropout_rate': uniform(0.1, 0.5),
            'learning_rate': uniform(0.0001, 0.01),
            'bidirectional': [True, False],
            'batch_norm': [True, False]
        }
        
        # Create HalvingRandomSearchCV
        search = HalvingRandomSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            factor=3,  # Reduce candidates by a factor of 3 in each iteration
            resource='epochs',  # Resource to increase with iterations
            max_resources=20,  # Maximum number of epochs
            min_resources=5,   # Start with 5 epochs
            cv=3,              # 3-fold cross-validation
            n_jobs=-1,         # Use all available cores
            verbose=2,
            random_state=42,
            error_score='raise',
            return_train_score=True
        )
        
        # Combine training and validation data for cross-validation
        X_combined = np.vstack((X_train_seq, X_val_seq))
        y_combined = np.vstack((y_train_seq, y_val_seq))
        
        # Fit the search
        search.fit(X_combined, y_combined)
        
        # Get best parameters
        best_params = search.best_params_
        print("\nBest hyperparameters:")
        for param, value in best_params.items():
            print(f"{param}: {value}")
        
        # Update config with best parameters
        self.config['lstm_units'] = [
            best_params['lstm_units_1'],
            best_params['lstm_units_2'],
            best_params['lstm_units_3']
        ]
        self.config['dense_units'] = [
            best_params['dense_units_1'],
            best_params['dense_units_2']
        ]
        self.config['dropout_rates'] = [best_params['dropout_rate']] * 5  # Use same dropout rate for all layers
        self.config['learning_rate'] = best_params['learning_rate']
        self.config['bidirectional'] = best_params['bidirectional']
        self.config['batch_norm'] = best_params['batch_norm']
        
        # Save hyperparameter search results
        cv_results_df = pd.DataFrame(search.cv_results_)
        cv_results_df.to_csv(f'results/lstm_lowres_improved/hyperparameter_search_{self.timestamp}.csv', index=False)
        
        # Plot hyperparameter search results
        self.plot_hyperparameter_search_results(search)
        
        return best_params
    
    def plot_hyperparameter_search_results(self, search):
        """
        Plot hyperparameter search results.
        
        Args:
            search: Fitted HalvingRandomSearchCV object
        """
        # Create directory for plots
        os.makedirs('results/lstm_lowres_improved/hyperparameter_search', exist_ok=True)
        
        # Extract results
        results = pd.DataFrame(search.cv_results_)
        
        # Plot iterations
        plt.figure(figsize=(10, 6))
        plt.plot(results['iter'], -results['mean_test_score'], 'o-')
        plt.xlabel('Iteration')
        plt.ylabel('Mean Negative MSE')
        plt.title('Performance by Iteration')
        plt.grid(True)
        plt.savefig(f'results/lstm_lowres_improved/hyperparameter_search/iterations_{self.timestamp}.png')
        
        # Plot number of candidates by iteration
        plt.figure(figsize=(10, 6))
        iterations = results['iter'].unique()
        n_candidates = [results[results['iter'] == i].shape[0] for i in iterations]
        plt.bar(iterations, n_candidates)
        plt.xlabel('Iteration')
        plt.ylabel('Number of Candidates')
        plt.title('Number of Candidates by Iteration')
        plt.grid(True, axis='y')
        plt.savefig(f'results/lstm_lowres_improved/hyperparameter_search/candidates_{self.timestamp}.png')
        
        # Plot learning rate vs score
        plt.figure(figsize=(10, 6))
        plt.scatter(results['param_learning_rate'], -results['mean_test_score'], alpha=0.7)
        plt.xlabel('Learning Rate')
        plt.ylabel('Mean Negative MSE')
        plt.title('Learning Rate vs Performance')
        plt.grid(True)
        plt.savefig(f'results/lstm_lowres_improved/hyperparameter_search/learning_rate_{self.timestamp}.png')
        
        # Plot dropout rate vs score
        plt.figure(figsize=(10, 6))
        plt.scatter(results['param_dropout_rate'], -results['mean_test_score'], alpha=0.7)
        plt.xlabel('Dropout Rate')
        plt.ylabel('Mean Negative MSE')
        plt.title('Dropout Rate vs Performance')
        plt.grid(True)
        plt.savefig(f'results/lstm_lowres_improved/hyperparameter_search/dropout_rate_{self.timestamp}.png')
        
        print(f"Hyperparameter search plots saved to results/lstm_lowres_improved/hyperparameter_search/")
    
    def build_model(self, input_shape):
        """
        Build an enhanced LSTM model with bidirectional layers and batch normalization.
        
        Args:
            input_shape: Shape of input data (time_steps, features)
            
        Returns:
            Compiled LSTM model
        """
        model = Sequential()
        
        # First LSTM layer
        if self.config['bidirectional']:
            model.add(Bidirectional(
                LSTM(units=self.config['lstm_units'][0], return_sequences=True),
                input_shape=input_shape
            ))
        else:
            model.add(LSTM(
                units=self.config['lstm_units'][0],
                return_sequences=True,
                input_shape=input_shape
            ))
        
        if self.config['batch_norm']:
            model.add(BatchNormalization())
        
        model.add(Dropout(self.config['dropout_rates'][0]))
        
        # Middle LSTM layers
        for i in range(1, len(self.config['lstm_units']) - 1):
            if self.config['bidirectional']:
                model.add(Bidirectional(
                    LSTM(units=self.config['lstm_units'][i], return_sequences=True)
                ))
            else:
                model.add(LSTM(
                    units=self.config['lstm_units'][i],
                    return_sequences=True
                ))
            
            if self.config['batch_norm']:
                model.add(BatchNormalization())
            
            model.add(Dropout(self.config['dropout_rates'][min(i, len(self.config['dropout_rates'])-1)]))
        
        # Last LSTM layer
        if self.config['bidirectional']:
            model.add(Bidirectional(
                LSTM(units=self.config['lstm_units'][-1], return_sequences=False)
            ))
        else:
            model.add(LSTM(
                units=self.config['lstm_units'][-1],
                return_sequences=False
            ))
        
        if self.config['batch_norm']:
            model.add(BatchNormalization())
        
        model.add(Dropout(self.config['dropout_rates'][min(len(self.config['lstm_units'])-1, len(self.config['dropout_rates'])-1)]))
        
        # Dense layers
        for i, units in enumerate(self.config['dense_units']):
            model.add(Dense(units, activation='relu'))
            
            if self.config['batch_norm']:
                model.add(BatchNormalization())
            
            if i < len(self.config['dense_units']) - 1:  # No dropout after the last dense layer
                model.add(Dropout(self.config['dropout_rates'][min(len(self.config['lstm_units']) + i, len(self.config['dropout_rates'])-1)]))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def create_callbacks(self):
        """
        Create callbacks for model training.
        
        Returns:
            List of callbacks
        """
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        model_checkpoint = ModelCheckpoint(
            filepath=f'models/lstm_lowres_improved/model_{self.timestamp}.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        return [early_stopping, model_checkpoint, reduce_lr]
    
    def evaluate_model(self, model, X_test_seq, y_test, target_scaler):
        """
        Evaluate the model and return metrics.
        
        Args:
            model: Trained LSTM model
            X_test_seq: Test sequences
            y_test: True target values (scaled)
            target_scaler: Scaler for the target variable
            
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
        
        print("\nModel Evaluation (Scaled):")
        print(f"Mean Squared Error (MSE): {mse_scaled:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse_scaled:.4f}")
        print(f"Mean Absolute Error (MAE): {mae_scaled:.4f}")
        print(f"R² Score: {r2_scaled:.4f}")
        
        # Inverse transform predictions and true values
        y_test_inv = target_scaler.inverse_transform(y_test)
        y_pred_inv = target_scaler.inverse_transform(y_pred)
        
        # Calculate metrics on original scale
        mse = mean_squared_error(y_test_inv, y_pred_inv)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        r2 = r2_score(y_test_inv, y_pred_inv)
        
        print("\nModel Evaluation (Original Scale):")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mse_scaled': mse_scaled,
            'rmse_scaled': rmse_scaled,
            'mae_scaled': mae_scaled,
            'r2_scaled': r2_scaled,
            'y_test_inv': y_test_inv,
            'y_pred_inv': y_pred_inv,
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
        plt.savefig(f'results/lstm_lowres_improved/history_{self.timestamp}.png')
        
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
        
        plt.savefig(f'results/lstm_lowres_improved/predictions_scaled_{self.timestamp}.png')
        
        # Plot predictions vs actual (original scale)
        plt.figure(figsize=(14, 7))
        
        plt.plot(indices, evaluation_results['y_test_inv'][:sample_size], 'b-', label='Actual Power Output')
        plt.plot(indices, evaluation_results['y_pred_inv'][:sample_size], 'r-', label='Predicted Power Output')
        plt.title('Actual vs Predicted PV Power')
        plt.xlabel('Time Steps')
        plt.ylabel('Power Output (W)')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(f'results/lstm_lowres_improved/predictions_{self.timestamp}.png')
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(evaluation_results['y_test_inv'], evaluation_results['y_pred_inv'], alpha=0.5)
        plt.plot([evaluation_results['y_test_inv'].min(), evaluation_results['y_test_inv'].max()], 
                 [evaluation_results['y_test_inv'].min(), evaluation_results['y_test_inv'].max()], 'r--')
        plt.title('Actual vs Predicted')
        plt.xlabel('Actual Power Output (W)')
        plt.ylabel('Predicted Power Output (W)')
        plt.grid(True)
        
        plt.savefig(f'results/lstm_lowres_improved/scatter_{self.timestamp}.png')
    
    def run_pipeline(self, optimize_hyperparams=True):
        """
        Run the full LSTM forecasting pipeline for low-resolution data.
        
        Args:
            optimize_hyperparams: Whether to perform hyperparameter optimization
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("Running improved LSTM forecasting pipeline for low-resolution (1-hour) data")
        
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
                X_train_seq, y_train_seq, X_val_seq, y_val_seq
            )
            print(f"Optimized hyperparameters: {best_params}")
            
            # Save optimized hyperparameters
            with open(f'models/lstm_lowres_improved/best_params_{self.timestamp}.txt', 'w') as f:
                for param, value in best_params.items():
                    f.write(f"{param}: {value}\n")
        
        # Build model with optimized hyperparameters
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        model = self.build_model(input_shape)
        model.summary()
        
        # Create callbacks
        callbacks = self.create_callbacks()
        
        # Train model
        print("\nTraining model with optimized hyperparameters...")
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
        model.save(f'models/lstm_lowres_improved/final_model_{self.timestamp}.keras')
        print(f"Model saved to models/lstm_lowres_improved/final_model_{self.timestamp}.keras")
        
        return evaluation_results


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run improved LSTM model with hyperparameter optimization')
    parser.add_argument('--optimize', action='store_true', help='Perform hyperparameter optimization')
    parser.add_argument('--sequence_length', type=int, default=48, help='Sequence length (hours to look back)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of epochs for training')
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
    
    print(f"Running improved LSTM model with parameters:")
    print(f"- Sequence length: {SEQUENCE_LENGTH}")
    print(f"- Batch size: {BATCH_SIZE}")
    print(f"- Max epochs: {EPOCHS}")
    print(f"- Hyperparameter optimization: {'Enabled' if OPTIMIZE else 'Disabled'}")
    
    # Initialize and run forecaster
    forecaster = LSTMLowResImproved(
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )
    
    # Run pipeline
    metrics = forecaster.run_pipeline(optimize_hyperparams=OPTIMIZE)
    
    # Print final results
    print("\nFinal Results:")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"R²: {metrics['r2']:.4f}")
    print(f"R² (Scaled): {metrics['r2_scaled']:.4f}")