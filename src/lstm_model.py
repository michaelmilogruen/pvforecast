import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import joblib
from datetime import datetime

class LSTMForecaster:
    def __init__(self, sequence_length=24, batch_size=32, epochs=50):
        """
        Initialize the LSTM forecaster with configuration.
        
        Args:
            sequence_length: Number of time steps to look back for prediction
            batch_size: Batch size for training
            epochs: Maximum number of epochs for training
        """
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.models = {}
        self.histories = {}
        
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
    
    def build_lstm_model(self, input_shape):
        """
        Build LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            
        Returns:
            Compiled LSTM model
        """
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def train_model(self, prepared_data, feature_set_name):
        """
        Train LSTM model on prepared data.
        
        Args:
            prepared_data: Dictionary with prepared X and y data
            feature_set_name: Name of the feature set for model identification
            
        Returns:
            Trained model and training history
        """
        print(f"\nTraining model for {feature_set_name} feature set...")
        
        # Get input shape from training data
        input_shape = (prepared_data['train']['X'].shape[1], prepared_data['train']['X'].shape[2])
        
        # Build model
        model = self.build_lstm_model(input_shape)
        
        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(f'models/lstm/{feature_set_name}_model.keras', save_best_only=True)
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
    
    def plot_training_history(self, feature_set_name):
        """
        Plot training history.
        
        Args:
            feature_set_name: Name of the feature set for plot title
        """
        history = self.histories[feature_set_name]
        
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title(f'Loss - {feature_set_name} Model')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        
        # Plot MAE
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Train')
        plt.plot(history.history['val_mae'], label='Validation')
        plt.title(f'Mean Absolute Error - {feature_set_name} Model')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'results/{feature_set_name}_history.png')
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
    
    def run_pipeline(self, data_filepath, target_col='power_w'):
        """
        Run the complete LSTM forecasting pipeline.
        
        Args:
            data_filepath: Path to the parquet file with data
            target_col: Target column name
        """
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
            
            # Train model
            model, history = self.train_model(prepared_data, set_name)
            
            # Plot training history
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
    EPOCHS = 100
    
    # Initialize forecaster
    forecaster = LSTMForecaster(
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )
    
    # Run pipeline
    metrics = forecaster.run_pipeline('data/processed_training_data.parquet')
    
    # Print final comparison
    print("\nModel Comparison:")
    for set_name, metrics in metrics.items():
        print(f"{set_name} model - RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.4f}")