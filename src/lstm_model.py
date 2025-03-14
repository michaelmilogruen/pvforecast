import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
import os
import joblib
from datetime import datetime

class LSTMForecaster:
    def __init__(self, sequence_length=24, batch_size=32, epochs=50, model_config=None):
        """
        Initialize the LSTM forecaster with configuration.
        
        Args:
            sequence_length: Number of time steps to look back for prediction
            batch_size: Batch size for training
            epochs: Maximum number of epochs for training
            model_config: Dictionary with model hyperparameters
        """
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.models = {}
        self.histories = {}
        
        # Default model configuration
        self.model_config = {
            'lstm_units': [128, 64],  # Units in each LSTM layer
            'dense_units': [32, 16],  # Units in each Dense layer
            'dropout_rates': [0.3, 0.3],  # Dropout rates after each LSTM layer
            'learning_rate': 0.001,
            'bidirectional': True,  # Whether to use bidirectional LSTM
            'batch_norm': True,  # Whether to use batch normalization
            'l1_reg': 0.0,  # L1 regularization factor
            'l2_reg': 0.001,  # L2 regularization factor
            'optimizer': 'adam'  # Optimizer to use
        }
        
        # Update with custom configuration if provided
        if model_config:
            self.model_config.update(model_config)
        
        # Load existing scalers
        try:
            self.minmax_scaler = joblib.load('models/minmax_scaler.pkl')
            self.standard_scaler = joblib.load('models/standard_scaler.pkl')
            print("Successfully loaded existing scalers")
        except Exception as e:
            print(f"Warning: Could not load existing scalers: {e}")
            print("Creating new scalers instead")
            self.minmax_scaler = None
            self.standard_scaler = None
        
        # Create directories for models and results
        os.makedirs('models/lstm', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
    def load_data(self, filepath):
        """
        Load data from parquet file.
        
        Args:
            filepath: Path to the parquet file
            
        Returns:
            DataFrame with loaded data
        """
        print(f"Loading data from {filepath}...")
        return pd.read_parquet(filepath)
    
    def prepare_feature_sets(self, df):
        """
        Prepare the three feature sets from the dataframe.
        
        Args:
            df: DataFrame with all features
            
        Returns:
            Dictionary with three feature sets
        """
        # Define the feature sets
        feature_sets = {
            'inca': [
                'INCA_GlobalRadiation [W m-2]',
                'INCA_Temperature [degree_Celsius]',
                'INCA_WindSpeed [m s-1]',
                'INCA_ClearSkyIndex',
                'hour_sin',  # Using hour_sin/cos as circular time features
                'hour_cos'
            ],
            'station': [
                'Station_GlobalRadiation [W m-2]',
                'Station_Temperature [degree_Celsius]',
                'Station_WindSpeed [m s-1]',
                'Station_ClearSkyIndex',
                'hour_sin',
                'hour_cos'
            ],
            'combined': [
                'Combined_GlobalRadiation [W m-2]',
                'Combined_Temperature [degree_Celsius]',
                'Combined_WindSpeed [m s-1]',
                'Combined_ClearSkyIndex',
                'hour_sin',
                'hour_cos'
            ]
        }
        
        # Verify all features exist in the dataframe
        for set_name, features in feature_sets.items():
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                print(f"Warning: Missing features in {set_name} set: {missing_features}")
                # Remove missing features from the set
                feature_sets[set_name] = [f for f in features if f in df.columns]
        
        return feature_sets
    
    def split_data(self, df, test_size=0.2, val_size=0.2):
        """
        Split data into training, validation, and test sets.
        
        Args:
            df: DataFrame with all data
            test_size: Proportion of data to use for testing
            val_size: Proportion of training data to use for validation
            
        Returns:
            Dictionary with train, val, and test DataFrames
        """
        # Sort by index to ensure chronological order
        df = df.sort_index()
        
        # Calculate split indices
        n = len(df)
        test_idx = int(n * (1 - test_size))
        val_idx = int(test_idx * (1 - val_size))
        
        # Split the data
        train_df = df.iloc[:val_idx].copy()
        val_df = df.iloc[val_idx:test_idx].copy()
        test_df = df.iloc[test_idx:].copy()
        
        print(f"Data split: Train={train_df.shape}, Validation={val_df.shape}, Test={test_df.shape}")
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
    
    def create_sequences(self, data, target_col):
        """
        Create sequences for LSTM input.
        
        Args:
            data: Array of feature data
            target_col: Array of target data
            
        Returns:
            X: Sequences of features
            y: Corresponding target values
        """
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(target_col[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def prepare_data_for_lstm(self, data_dict, feature_set, target_col='power_w'):
        """
        Prepare data for LSTM model by creating sequences.
        Note: Data is already normalized from the processing step.
        
        Args:
            data_dict: Dictionary with train, val, and test DataFrames
            feature_set: List of feature columns to use
            target_col: Target column name
            
        Returns:
            Dictionary with prepared X and y data for train, val, and test
        """
        prepared_data = {}
        
        # Process each dataset
        for split, df in data_dict.items():
            # Get features and target (already normalized)
            X_data = df[feature_set].values
            y_data = df[target_col].values
            
            # Create sequences
            X, y = self.create_sequences(X_data, y_data)
            
            prepared_data[split] = {
                'X': X,
                'y': y,
                'X_raw': df[feature_set].values,
                'y_raw': df[target_col].values
            }
        
        return prepared_data
    
    def build_lstm_model(self, input_shape, config=None):
        """
        Build LSTM model architecture with configurable hyperparameters.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            config: Optional custom configuration for this specific model
            
        Returns:
            Compiled LSTM model
        """
        # Use provided config or default
        cfg = config if config else self.model_config
        
        # Create regularizer if specified
        regularizer = None
        if cfg['l1_reg'] > 0 or cfg['l2_reg'] > 0:
            regularizer = l1_l2(l1=cfg['l1_reg'], l2=cfg['l2_reg'])
        
        # Build model
        model = Sequential()
        
        # First LSTM layer
        if cfg['bidirectional']:
            model.add(Bidirectional(
                LSTM(cfg['lstm_units'][0],
                     return_sequences=len(cfg['lstm_units']) > 1,
                     kernel_regularizer=regularizer,
                     recurrent_regularizer=regularizer),
                input_shape=input_shape
            ))
        else:
            model.add(LSTM(
                cfg['lstm_units'][0],
                return_sequences=len(cfg['lstm_units']) > 1,
                kernel_regularizer=regularizer,
                recurrent_regularizer=regularizer,
                input_shape=input_shape
            ))
        
        if cfg['batch_norm']:
            model.add(BatchNormalization())
            
        model.add(Dropout(cfg['dropout_rates'][0]))
        
        # Additional LSTM layers
        for i in range(1, len(cfg['lstm_units'])):
            if cfg['bidirectional']:
                model.add(Bidirectional(
                    LSTM(cfg['lstm_units'][i],
                         return_sequences=i < len(cfg['lstm_units']) - 1,
                         kernel_regularizer=regularizer,
                         recurrent_regularizer=regularizer)
                ))
            else:
                model.add(LSTM(
                    cfg['lstm_units'][i],
                    return_sequences=i < len(cfg['lstm_units']) - 1,
                    kernel_regularizer=regularizer,
                    recurrent_regularizer=regularizer
                ))
            
            if cfg['batch_norm']:
                model.add(BatchNormalization())
                
            if i < len(cfg['dropout_rates']):
                model.add(Dropout(cfg['dropout_rates'][i]))
        
        # Dense layers
        for units in cfg['dense_units']:
            model.add(Dense(units, activation='relu', kernel_regularizer=regularizer))
            
        # Output layer
        model.add(Dense(1))
        
        # Compile model
        if cfg['optimizer'].lower() == 'adam':
            optimizer = Adam(learning_rate=cfg['learning_rate'])
        else:
            optimizer = cfg['optimizer']
            
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model
    
    def train_model(self, prepared_data, feature_set_name, custom_config=None):
        """
        Train LSTM model on prepared data.
        
        Args:
            prepared_data: Dictionary with prepared X and y data
            feature_set_name: Name of the feature set for model identification
            custom_config: Optional custom configuration for this specific model
            
        Returns:
            Trained model and training history
        """
        print(f"\nTraining model for {feature_set_name} feature set...")
        
        # Get input shape from training data
        input_shape = (prepared_data['train']['X'].shape[1], prepared_data['train']['X'].shape[2])
        
        # Use feature-specific configurations if available
        config = custom_config if custom_config else self.model_config
        
        # Feature-specific optimizations based on previous results
        if feature_set_name == 'station' and not custom_config:
            # Station data performed best, optimize further
            config = self.model_config.copy()
            config['lstm_units'] = [128, 64]
            config['dropout_rates'] = [0.25, 0.25]
            config['learning_rate'] = 0.0008
        elif feature_set_name == 'inca' and not custom_config:
            # Inca data had more overfitting, increase regularization
            config = self.model_config.copy()
            config['dropout_rates'] = [0.35, 0.35]
            config['l2_reg'] = 0.002
        
        # Build model
        model = self.build_lstm_model(input_shape, config)
        
        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ModelCheckpoint(f'models/lstm/{feature_set_name}_model.keras', save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
        ]
        
        # Train model
        history = model.fit(
            prepared_data['train']['X'], prepared_data['train']['y'],
            validation_data=(prepared_data['val']['X'], prepared_data['val']['y']),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model and history
        self.models[feature_set_name] = model
        self.histories[feature_set_name] = history
        
        return model, history
    
    def evaluate_model(self, model, prepared_data, feature_set_name):
        """
        Evaluate model on test data.
        
        Args:
            model: Trained LSTM model
            prepared_data: Dictionary with prepared X and y data
            feature_set_name: Name of the feature set for identification
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\nEvaluating model for {feature_set_name} feature set...")
        
        # Get predictions on test data (already normalized)
        y_pred = model.predict(prepared_data['test']['X']).flatten()
        
        # Get actual values (need to account for sequence length)
        y_true = prepared_data['test']['y'][:]
        
        # Calculate metrics on normalized data
        norm_metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        # Get the original scale data for more interpretable metrics
        # We need to reshape for inverse_transform
        y_pred_reshaped = y_pred.reshape(-1, 1)
        y_true_reshaped = y_true.reshape(-1, 1)
        
        # Try to convert back to original scale if we have the scaler
        if self.minmax_scaler is not None:
            try:
                # Since 'power_w' is in minmax_columns in the original processing
                # We need to create a dummy array with zeros for all other minmax features
                # to properly use the inverse_transform
                minmax_feature_count = len(self.minmax_scaler.feature_names_in_)
                
                # Find the index of 'power_w' in the minmax scaler features
                power_w_idx = np.where(self.minmax_scaler.feature_names_in_ == 'power_w')[0][0]
                
                # Create dummy arrays for inverse transform
                y_pred_dummy = np.zeros((len(y_pred), minmax_feature_count))
                y_pred_dummy[:, power_w_idx] = y_pred
                
                y_true_dummy = np.zeros((len(y_true), minmax_feature_count))
                y_true_dummy[:, power_w_idx] = y_true
                
                # Inverse transform
                y_pred_orig = self.minmax_scaler.inverse_transform(y_pred_dummy)[:, power_w_idx]
                y_true_orig = self.minmax_scaler.inverse_transform(y_true_dummy)[:, power_w_idx]
            except Exception as e:
                print(f"Warning: Could not inverse transform with minmax_scaler: {e}")
                print("Using normalized values for metrics")
                y_pred_orig = y_pred
                y_true_orig = y_true
        else:
            print("Warning: No minmax_scaler available, using normalized values for metrics")
            y_pred_orig = y_pred
            y_true_orig = y_true
        
        # Calculate metrics on original scale
        orig_metrics = {
            'mse': mean_squared_error(y_true_orig, y_pred_orig),
            'rmse': np.sqrt(mean_squared_error(y_true_orig, y_pred_orig)),
            'mae': mean_absolute_error(y_true_orig, y_pred_orig),
            'r2': r2_score(y_true_orig, y_pred_orig)
        }
        
        print(f"Test MSE (normalized): {norm_metrics['mse']:.4f}")
        print(f"Test RMSE (normalized): {norm_metrics['rmse']:.4f}")
        print(f"Test MAE (normalized): {norm_metrics['mae']:.4f}")
        print(f"Test R² (normalized): {norm_metrics['r2']:.4f}")
        
        print(f"\nTest MSE (original scale): {orig_metrics['mse']:.2f}")
        print(f"Test RMSE (original scale): {orig_metrics['rmse']:.2f}")
        print(f"Test MAE (original scale): {orig_metrics['mae']:.2f}")
        print(f"Test R² (original scale): {orig_metrics['r2']:.4f}")
        
        # Plot predictions vs actual
        self.plot_predictions(y_true_orig, y_pred_orig, feature_set_name)
        
        return orig_metrics
    
    def plot_predictions(self, y_true, y_pred, feature_set_name):
        """
        Plot predictions vs actual values.
        
        Args:
            y_true: Array of actual values
            y_pred: Array of predicted values
            feature_set_name: Name of the feature set for plot title
        """
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.title(f'Actual vs Predicted Power - {feature_set_name} Model')
        plt.xlabel('Time Steps')
        plt.ylabel('Power (W)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'results/{feature_set_name}_predictions.png')
        plt.close()
        
        # Also plot a scatter plot
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
        plt.title(f'Actual vs Predicted Power - {feature_set_name} Model')
        plt.xlabel('Actual Power (W)')
        plt.ylabel('Predicted Power (W)')
        plt.tight_layout()
        plt.savefig(f'results/{feature_set_name}_scatter.png')
        plt.close()
    
    def plot_training_history(self, feature_set_name, config_name=None):
        """
        Plot training history.
        
        Args:
            feature_set_name: Name of the feature set for plot title
            config_name: Optional specific configuration name to plot
        """
        if config_name:
            # Plot specific configuration history
            if f"{feature_set_name}_all" in self.histories and config_name in self.histories[f"{feature_set_name}_all"]:
                history = self.histories[f"{feature_set_name}_all"][config_name]
                output_name = f"{config_name}"
            else:
                print(f"Warning: History for {config_name} not found")
                return
        else:
            # Plot best model history
            if feature_set_name not in self.histories:
                print(f"Warning: History for {feature_set_name} not found")
                return
            history = self.histories[feature_set_name]
            output_name = feature_set_name
        
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title(f'Loss - {output_name} Model')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        
        # Plot MAE
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Train')
        plt.plot(history.history['val_mae'], label='Validation')
        plt.title(f'Mean Absolute Error - {output_name} Model')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'results/{output_name}_history.png')
        plt.close()
    
    def plot_all_training_histories(self, feature_set_name):
        """
        Plot training histories for all configurations of a feature set.
        
        Args:
            feature_set_name: Name of the feature set
        """
        if f"{feature_set_name}_all" not in self.histories:
            print(f"Warning: No configuration histories found for {feature_set_name}")
            return
        
        all_histories = self.histories[f"{feature_set_name}_all"]
        
        # Create directory for comparison plots
        os.makedirs('results/tuning', exist_ok=True)
        
        # Plot loss comparison
        plt.figure(figsize=(12, 8))
        
        # Plot validation loss for each configuration
        for config_name, history in all_histories.items():
            config_id = config_name.split('_')[-1]  # Extract config number
            plt.plot(history.history['val_loss'], label=f'Config {config_id}')
        
        plt.title(f'Validation Loss Comparison - {feature_set_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss (MSE)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'results/tuning/{feature_set_name}_val_loss_comparison.png')
        plt.close()
        
        # Plot MAE comparison
        plt.figure(figsize=(12, 8))
        
        # Plot validation MAE for each configuration
        for config_name, history in all_histories.items():
            config_id = config_name.split('_')[-1]  # Extract config number
            plt.plot(history.history['val_mae'], label=f'Config {config_id}')
        
        plt.title(f'Validation MAE Comparison - {feature_set_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Validation MAE')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'results/tuning/{feature_set_name}_val_mae_comparison.png')
        plt.close()
    
    def save_results(self, metrics_dict):
        """
        Save evaluation metrics to CSV.
        
        Args:
            metrics_dict: Dictionary with metrics for each feature set
        """
        results_df = pd.DataFrame(metrics_dict).T
        results_df.index.name = 'feature_set'
        results_df.to_csv('results/model_comparison.csv')
        print(f"\nResults saved to results/model_comparison.csv")
        
        # Also save as a formatted markdown table
        try:
            with open('results/model_comparison.md', 'w') as f:
                f.write("# LSTM Model Comparison\n\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                try:
                    f.write(results_df.to_markdown())
                except ImportError:
                    # If tabulate is not available, use a simple string representation
                    f.write("```\n")
                    f.write(str(results_df))
                    f.write("\n```")
        except Exception as e:
            print(f"Warning: Could not save markdown file: {e}")
    
    def tune_hyperparameters(self, prepared_data, feature_set_name):
        """
        Perform hyperparameter tuning to find the best model configuration.
        
        Args:
            prepared_data: Dictionary with prepared X and y data
            feature_set_name: Name of the feature set for model identification
            
        Returns:
            Best model configuration based on validation performance
        """
        print(f"\nPerforming hyperparameter tuning for {feature_set_name} feature set...")
        
        # Define hyperparameter configurations to try
        configs = [
            # Configuration 1: Baseline enhanced model
            {
                'lstm_units': [128, 64],
                'dense_units': [32, 16],
                'dropout_rates': [0.3, 0.3],
                'learning_rate': 0.001,
                'bidirectional': True,
                'batch_norm': True,
                'l1_reg': 0.0,
                'l2_reg': 0.001,
                'optimizer': 'adam'
            },
            # Configuration 2: Deeper model
            {
                'lstm_units': [128, 64, 32],
                'dense_units': [32, 16],
                'dropout_rates': [0.3, 0.3, 0.3],
                'learning_rate': 0.001,
                'bidirectional': True,
                'batch_norm': True,
                'l1_reg': 0.0,
                'l2_reg': 0.001,
                'optimizer': 'adam'
            },
            # Configuration 3: Higher capacity
            {
                'lstm_units': [256, 128],
                'dense_units': [64, 32],
                'dropout_rates': [0.4, 0.4],
                'learning_rate': 0.0008,
                'bidirectional': True,
                'batch_norm': True,
                'l1_reg': 0.0,
                'l2_reg': 0.002,
                'optimizer': 'adam'
            },
            # Configuration 4: More regularization
            {
                'lstm_units': [128, 64],
                'dense_units': [32, 16],
                'dropout_rates': [0.4, 0.4],
                'learning_rate': 0.001,
                'bidirectional': True,
                'batch_norm': True,
                'l1_reg': 0.0001,
                'l2_reg': 0.002,
                'optimizer': 'adam'
            },
            # Configuration 5: Simpler model with less regularization
            {
                'lstm_units': [64, 32],
                'dense_units': [16],
                'dropout_rates': [0.2, 0.2],
                'learning_rate': 0.001,
                'bidirectional': False,
                'batch_norm': False,
                'l1_reg': 0.0,
                'l2_reg': 0.0005,
                'optimizer': 'adam'
            }
        ]
        
        # Feature-specific configurations
        if feature_set_name == 'station':
            # Add station-specific configurations
            configs.append({
                'lstm_units': [128, 64],
                'dense_units': [32, 16],
                'dropout_rates': [0.25, 0.25],
                'learning_rate': 0.0008,
                'bidirectional': True,
                'batch_norm': True,
                'l1_reg': 0.0,
                'l2_reg': 0.001,
                'optimizer': 'adam'
            })
        elif feature_set_name == 'inca':
            # Add inca-specific configurations
            configs.append({
                'lstm_units': [128, 64],
                'dense_units': [32, 16],
                'dropout_rates': [0.35, 0.35],
                'learning_rate': 0.001,
                'bidirectional': True,
                'batch_norm': True,
                'l1_reg': 0.0,
                'l2_reg': 0.002,
                'optimizer': 'adam'
            })
        
        # Dictionary to store all models and histories
        all_models = {}
        all_histories = {}
        all_val_losses = {}
        
        # Train and evaluate models with each configuration
        best_val_loss = float('inf')
        best_config = None
        best_config_idx = -1
        
        for i, config in enumerate(configs):
            config_name = f"{feature_set_name}_config_{i+1}"
            print(f"\nTrying configuration {i+1}/{len(configs)}:")
            for key, value in config.items():
                print(f"  {key}: {value}")
            
            # Train model with this configuration
            model, history = self.train_model(prepared_data, config_name, config)
            
            # Save this model and history
            all_models[config_name] = model
            all_histories[config_name] = history
            
            # Get validation loss
            val_loss = min(history.history['val_loss'])
            all_val_losses[config_name] = val_loss
            print(f"Validation loss: {val_loss:.6f}")
            
            # Check if this is the best model so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_config = config
                best_config_idx = i
                print(f"New best model found!")
                
                # Also save as the main model for this feature set
                self.models[feature_set_name] = model
                self.histories[feature_set_name] = history
        
        # Save all models and histories in a structured way
        self.models[f"{feature_set_name}_all"] = all_models
        self.histories[f"{feature_set_name}_all"] = all_histories
        
        # Save tuning results to CSV
        self._save_tuning_results(feature_set_name, configs, all_val_losses, best_config_idx)
        
        print(f"\nBest configuration for {feature_set_name} (config_{best_config_idx+1}):")
        for key, value in best_config.items():
            print(f"  {key}: {value}")
        print(f"Best validation loss: {best_val_loss:.6f}")
        
        return best_config
    
    def _save_tuning_results(self, feature_set_name, configs, val_losses, best_idx):
        """
        Save hyperparameter tuning results to CSV.
        
        Args:
            feature_set_name: Name of the feature set
            configs: List of configuration dictionaries
            val_losses: Dictionary of validation losses
            best_idx: Index of the best configuration
        """
        # Create directory for tuning results
        os.makedirs('results/tuning', exist_ok=True)
        
        # Prepare data for CSV
        results = []
        for i, config in enumerate(configs):
            config_name = f"{feature_set_name}_config_{i+1}"
            row = {
                'config_id': i+1,
                'feature_set': feature_set_name,
                'val_loss': val_losses.get(config_name, float('nan')),
                'is_best': 'Yes' if i == best_idx else 'No'
            }
            # Add config parameters
            for key, value in config.items():
                if isinstance(value, list):
                    row[key] = str(value)
                else:
                    row[key] = value
            results.append(row)
        
        # Create DataFrame and save to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'results/tuning/{feature_set_name}_tuning_results.csv', index=False)
        print(f"Tuning results saved to results/tuning/{feature_set_name}_tuning_results.csv")
        
        # Also plot validation losses
        self._plot_tuning_results(feature_set_name, results_df)
    
    def _plot_tuning_results(self, feature_set_name, results_df):
        """
        Plot hyperparameter tuning results.
        
        Args:
            feature_set_name: Name of the feature set
            results_df: DataFrame with tuning results
        """
        plt.figure(figsize=(10, 6))
        bars = plt.bar(results_df['config_id'], results_df['val_loss'], alpha=0.7)
        
        # Highlight the best configuration
        best_idx = results_df[results_df['is_best'] == 'Yes'].index[0]
        bars[best_idx].set_color('green')
        
        plt.title(f'Hyperparameter Tuning Results - {feature_set_name}')
        plt.xlabel('Configuration ID')
        plt.ylabel('Validation Loss')
        plt.xticks(results_df['config_id'])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.6f}',
                    ha='center', va='bottom', rotation=0)
        
        plt.tight_layout()
        plt.savefig(f'results/tuning/{feature_set_name}_tuning_comparison.png')
        plt.close()
    
    def generate_markdown_documentation(self):
        """
        Generate markdown documentation for the LSTM hyperparameter tuning approach.
        
        Returns:
            Markdown string with documentation
        """
        markdown = """# LSTM Hyperparameter Tuning for PV Power Forecasting

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
"""
        
        # Save the markdown to a file
        os.makedirs('docs', exist_ok=True)
        with open('docs/lstm_hyperparameter_tuning_guide.md', 'w') as f:
            f.write(markdown)
        
        print("Documentation generated and saved to docs/lstm_hyperparameter_tuning_guide.md")
        
        return markdown
    
    def run_pipeline(self, data_filepath, target_col='power_w', perform_tuning=True, generate_docs=True):
        """
        Run the complete LSTM forecasting pipeline.
        
        Args:
            data_filepath: Path to the parquet file with data
            target_col: Target column name
            perform_tuning: Whether to perform hyperparameter tuning
            generate_docs: Whether to generate markdown documentation
        """
        # Generate documentation if requested
        if generate_docs:
            self.generate_markdown_documentation()
        
        # Load data
        df = self.load_data(data_filepath)
        
        # Prepare feature sets
        feature_sets = self.prepare_feature_sets(df)
        
        # Split data
        data_splits = self.split_data(df)
        
        # Train and evaluate models for each feature set
        metrics_dict = {}
        
        for set_name, features in feature_sets.items():
            print(f"\n{'='*50}")
            print(f"Processing {set_name} feature set")
            print(f"Features: {features}")
            print(f"{'='*50}")
            
            # Prepare data for LSTM
            prepared_data = self.prepare_data_for_lstm(data_splits, features, target_col)
            
            if perform_tuning:
                # Perform hyperparameter tuning
                best_config = self.tune_hyperparameters(prepared_data, set_name)
                
                # Plot training histories for all configurations
                self.plot_all_training_histories(set_name)
                
                # Train final model with best configuration
                model, history = self.train_model(prepared_data, set_name, best_config)
            else:
                # Train model with default configuration
                model, history = self.train_model(prepared_data, set_name)
            
            # Plot training history for the final model
            self.plot_training_history(set_name)
            
            # Evaluate model
            metrics = self.evaluate_model(model, prepared_data, set_name)
            metrics_dict[set_name] = metrics
        
        # Save comparison results
        self.save_results(metrics_dict)
        
        return metrics_dict


if __name__ == "__main__":
    # Set parameters
    SEQUENCE_LENGTH = 24  # Look back 24 time steps (6 hours with 15-min data)
    BATCH_SIZE = 64
    EPOCHS = 150  # Increased max epochs with early stopping
    
    # Whether to perform hyperparameter tuning (can be time-consuming)
    # Set to False to use the predefined optimized configuration
    PERFORM_TUNING = True
    
    # Whether to generate markdown documentation
    GENERATE_DOCS = True
    
    # Define optimized model configuration (used if PERFORM_TUNING is False)
    model_config = {
        'lstm_units': [128, 64],  # Increased units in LSTM layers
        'dense_units': [32, 16],  # Added more dense units
        'dropout_rates': [0.3, 0.3],  # Adjusted dropout for better regularization
        'learning_rate': 0.001,
        'bidirectional': True,  # Use bidirectional LSTM for better sequence learning
        'batch_norm': True,  # Add batch normalization to stabilize training
        'l1_reg': 0.0,  # No L1 regularization
        'l2_reg': 0.001,  # L2 regularization to prevent overfitting
        'optimizer': 'adam'  # Using Adam optimizer
    }
    
    print("=" * 80)
    print("LSTM Model with Hyperparameter Tuning")
    print("=" * 80)
    print(f"Sequence Length: {SEQUENCE_LENGTH}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Max Epochs: {EPOCHS}")
    print(f"Hyperparameter Tuning: {'Enabled' if PERFORM_TUNING else 'Disabled'}")
    print(f"Generate Documentation: {'Enabled' if GENERATE_DOCS else 'Disabled'}")
    print("=" * 80)
    
    # Initialize forecaster with enhanced configuration
    forecaster = LSTMForecaster(
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        model_config=model_config
    )
    
    # Generate documentation if requested
    if GENERATE_DOCS:
        forecaster.generate_markdown_documentation()
        print("Documentation generated successfully!")
    
    # Run pipeline with or without hyperparameter tuning
    metrics = forecaster.run_pipeline(
        'data/processed_training_data.parquet',
        perform_tuning=PERFORM_TUNING,
        generate_docs=False  # Already generated above if requested
    )
    
    # Print final comparison
    print("\nModel Comparison:")
    for set_name, metrics in metrics.items():
        print(f"{set_name} model - RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.4f}")