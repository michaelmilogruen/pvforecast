
# %% [markdown]
# # LSTM Model for PV Power Forecasting
# 
# This notebook implements an LSTM (Long Short-Term Memory) neural network for forecasting photovoltaic power output.

# %% [markdown]
# ## Data Loading and Preprocessing

# %%
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.regularizers import l1_l2
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

# %%
# Load the processed training data
data_path = 'data/processed_training_data.parquet'
df = pd.read_parquet(data_path)

# Display the first few rows
print(f"Loaded data shape: {df.shape}")
df.head()

# %%
# Feature selection
# Define function to prepare feature sets

def prepare_feature_sets(df):
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
            'hour_cos',
            'day_cos',   # Added day_cos as requested
            'day_sin',   # Added day_sin as requested
            'isNight'    # Added isNight as requested
        ],
        'station': [
            'Station_GlobalRadiation [W m-2]',
            'Station_Temperature [degree_Celsius]',
            'Station_WindSpeed [m s-1]',
            'Station_ClearSkyIndex',
            'hour_sin',
            'hour_cos',
            'day_cos',   # Added day_cos as requested
            'day_sin',   # Added day_sin as requested
            'isNight'    # Added isNight as requested
        ],
        'combined': [
            'Combined_GlobalRadiation [W m-2]',
            'Combined_Temperature [degree_Celsius]',
            'Combined_WindSpeed [m s-1]',
            'Combined_ClearSkyIndex',
            'hour_sin',
            'hour_cos',
            'day_cos',   # Added day_cos as requested
            'day_sin',   # Added day_sin as requested
            'isNight'    # Added isNight as requested
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

# Get feature sets
feature_sets = prepare_feature_sets(df)

# Select the feature set to use (default to 'station')
feature_set_name = 'station'
features = feature_sets[feature_set_name]

# Separate features that need scaling from those that don't
time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'isNight']
features_to_scale = [f for f in features if f not in time_features]
additional_features = [f for f in features if f in time_features]

print(f"Using feature set: {feature_set_name}")
print(f"Features to scale: {features_to_scale}")
print(f"Additional features: {additional_features}")

target = 'power_w'  # Target variable (power output in watts)

# Check for missing values
print("Missing values in dataset:")
print(df[features + [target]].isna().sum())

# %%
# Data normalization
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

# Fit and transform only the features that need scaling
scaled_features = scaler_x.fit_transform(df[features_to_scale])

# Get the additional features that don't need scaling
# These features (day_sin, day_cos, isNight) are already appropriately normalized or binary
unscaled_features = df[additional_features].values

# Combine scaled and unscaled features to create the complete feature set for training
X = np.hstack((scaled_features, unscaled_features))

# Print feature dimensions to verify all features are included
print(f"Scaled features shape: {scaled_features.shape}")
print(f"Unscaled features shape: {unscaled_features.shape}")
print(f"Combined features shape: {X.shape}")

# Reshape target to 2D array for scaling
y = scaler_y.fit_transform(df[[target]])

# Save the scalers for later use
joblib.dump(scaler_x, 'models/scaler_x.pkl')
joblib.dump(scaler_y, 'models/scaler_y.pkl')

# %% [markdown]
# ## Time Series Data Preparation

# %%
def create_sequences(X, y, time_steps=24):
    """
    Create sequences of data for time series forecasting.
    
    Args:
        X: Features array
        y: Target array
        time_steps: Number of time steps to look back
        
    Returns:
        X_seq: Sequences of features
        y_seq: Corresponding target values
    """
    X_seq, y_seq = [], []
    
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i + time_steps])
        y_seq.append(y[i + time_steps])
        
    return np.array(X_seq), np.array(y_seq)

# %%
# Create sequences for LSTM
time_steps = 24  # Look back 24 time steps (e.g., hours)
X_seq, y_seq = create_sequences(X, y, time_steps)

print(f"Sequence shapes - X: {X_seq.shape}, y: {y_seq.shape}")

# %%
# Split data into training and testing sets
train_size = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:train_size], X_seq[train_size:]
y_train, y_test = y_seq[:train_size], y_seq[train_size:]

print(f"Training set shapes - X: {X_train.shape}, y: {y_train.shape}")
print(f"Testing set shapes - X: {X_test.shape}, y: {y_test.shape}")

# %% [markdown]
# ## LSTM Model Definition

# %%
from tensorflow.keras.layers import Bidirectional, BatchNormalization
from tensorflow.keras.regularizers import l1, l2

def build_lstm_model(input_shape, config=None):
    """
    Build an LSTM model for time series forecasting.
    
    Args:
        input_shape: Shape of input data (time_steps, features)
        config: Dictionary containing model configuration parameters
        
    Returns:
        model: Compiled LSTM model
    """
    if config is None:
        # Default configuration
        config = {
            'lstm_units': [50, 50],
            'dense_units': [50],
            'dropout_rates': [0.2, 0.2],
            'learning_rate': 0.001,
            'bidirectional': False,
            'batch_norm': False,
            'l1_reg': 0.0,
            'l2_reg': 0.0,
            'optimizer': 'adam'
        }
    
    model = Sequential()
    
    # Extract configuration parameters
    lstm_units = config['lstm_units']
    dense_units = config['dense_units']
    dropout_rates = config['dropout_rates']
    learning_rate = config['learning_rate']
    bidirectional = config.get('bidirectional', False)
    batch_norm = config.get('batch_norm', False)
    l1_regularization = config.get('l1_reg', 0.0)
    l2_regularization = config.get('l2_reg', 0.0)
    
    # Regularizer
    regularizer = None
    if l1_regularization > 0 or l2_regularization > 0:
        regularizer = l1_l2(l1=l1_regularization, l2=l2_regularization)
    
    # First LSTM layer
    if bidirectional:
        model.add(Bidirectional(
            LSTM(units=lstm_units[0], return_sequences=True if len(lstm_units) > 1 else False,
                 kernel_regularizer=regularizer),
            input_shape=input_shape))
    else:
        model.add(LSTM(units=lstm_units[0], return_sequences=True if len(lstm_units) > 1 else False,
                       kernel_regularizer=regularizer, input_shape=input_shape))
    
    if batch_norm:
        model.add(BatchNormalization())
    
    model.add(Dropout(dropout_rates[0]))
    
    # Middle LSTM layers (if any)
    for i in range(1, len(lstm_units) - 1):
        if bidirectional:
            model.add(Bidirectional(LSTM(units=lstm_units[i], return_sequences=True,
                                         kernel_regularizer=regularizer)))
        else:
            model.add(LSTM(units=lstm_units[i], return_sequences=True,
                           kernel_regularizer=regularizer))
        
        if batch_norm:
            model.add(BatchNormalization())
        
        model.add(Dropout(dropout_rates[min(i, len(dropout_rates) - 1)]))
    
    # Last LSTM layer (if more than one)
    if len(lstm_units) > 1:
        if bidirectional:
            model.add(Bidirectional(LSTM(units=lstm_units[-1], return_sequences=False,
                                         kernel_regularizer=regularizer)))
        else:
            model.add(LSTM(units=lstm_units[-1], return_sequences=False,
                           kernel_regularizer=regularizer))
        
        if batch_norm:
            model.add(BatchNormalization())
        
        model.add(Dropout(dropout_rates[min(len(lstm_units) - 1, len(dropout_rates) - 1)]))
    
    # Dense layers
    for i, units in enumerate(dense_units):
        model.add(Dense(units, activation='relu', kernel_regularizer=regularizer))
        
        if batch_norm:
            model.add(BatchNormalization())
        
        if i < len(dense_units) - 1:  # No dropout after the last dense layer
            model.add(Dropout(dropout_rates[min(len(lstm_units) + i, len(dropout_rates) - 1)]))
    
    # Output layer
    model.add(Dense(1))
    
    # Compile the model
    if config['optimizer'].lower() == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif config['optimizer'].lower() == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = config['optimizer']
    
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

# %%
# Define the specific configuration from the request
station_config_2 = {
    'config_id': 2,
    'feature_set': 'station',
    'lstm_units': [128, 64, 32],
    'dense_units': [32, 16],
    'dropout_rates': [0.3, 0.3, 0.3],
    'learning_rate': 0.001,
    'bidirectional': True,
    'batch_norm': True,
    'l1_reg': 0.0,
    'l2_reg': 0.001,
    'optimizer': 'adam'
}

# Build the LSTM model with the specified configuration
input_shape = (X_train.shape[1], X_train.shape[2])  # (time_steps, features)
model = build_lstm_model(input_shape, config=station_config_2)

# Display model summary
print(f"Model input shape: {input_shape} (time_steps, features)")
print(f"Features include: {features_to_scale + additional_features}")
print(f"Using configuration: {station_config_2['config_id']} - {station_config_2['feature_set']}")
model.summary()

# %% [markdown]
# ## Hyperparameter Tuning

# %%
# Define hyperparameters to tune
hyperparameters = {
    'lstm_units': [32, 50, 64, 128],
    'dropout_rate': [0.1, 0.2, 0.3],
    'batch_size': [16, 32, 64],
    'learning_rate': [0.001, 0.01]
}

# Function to build and evaluate model with specific hyperparameters
def evaluate_model(lstm_units, dropout_rate, batch_size, learning_rate):
    # Build model with specified hyperparameters
    model = Sequential()
    model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=lstm_units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    
    # Compile with specified learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    # Train with early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=20,  # Reduced epochs for faster tuning
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Evaluate on validation data
    val_loss = min(history.history['val_loss'])
    return val_loss, model

# %%
# Simple grid search implementation
# Note: For a more comprehensive approach, consider using libraries like Keras Tuner or scikit-learn's GridSearchCV

results = []

# Uncomment to run the grid search (warning: may take significant time)
'''
for units in hyperparameters['lstm_units']:
    for dropout in hyperparameters['dropout_rate']:
        for batch in hyperparameters['batch_size']:
            for lr in hyperparameters['learning_rate']:
                print(f"Testing: units={units}, dropout={dropout}, batch_size={batch}, lr={lr}")
                val_loss, _ = evaluate_model(units, dropout, batch, lr)
                results.append({
                    'lstm_units': units,
                    'dropout_rate': dropout,
                    'batch_size': batch,
                    'learning_rate': lr,
                    'val_loss': val_loss
                })
                
# Convert results to DataFrame and sort by validation loss
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('val_loss')
results_df.head(10)  # Show top 10 configurations
'''

# For demonstration, we'll use the following hyperparameters
best_params = {
    'lstm_units': 64,
    'dropout_rate': 0.2,
    'batch_size': 32,
    'learning_rate': 0.001
}

print("Using hyperparameters:")
for param, value in best_params.items():
    print(f"  {param}: {value}")

# %% [markdown]
# ## Model Training with Station Config 2

# %%
# Define callbacks for early stopping and model checkpointing
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Create directory if it doesn't exist
os.makedirs('models/lstm', exist_ok=True)
os.makedirs('logs', exist_ok=True)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='models/lstm/station_config_2_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Create a TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='logs/station_config_2',
    histogram_freq=1
)

# %%
# Train the model with the specified configuration
batch_size = 32
epochs = 50

print(f"Training model with configuration ID {station_config_2['config_id']} - {station_config_2['feature_set']}")
print(f"LSTM units: {station_config_2['lstm_units']}")
print(f"Dense units: {station_config_2['dense_units']}")
print(f"Dropout rates: {station_config_2['dropout_rates']}")
print(f"Bidirectional: {station_config_2['bidirectional']}")
print(f"Batch normalization: {station_config_2['batch_norm']}")
print(f"L2 regularization: {station_config_2['l2_reg']}")

history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.2,
    callbacks=[early_stopping, model_checkpoint, tensorboard_callback],
    verbose=1
)

# %% [markdown]
# ## Model Evaluation

# %%
# Plot training history
plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot MAE
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.savefig('results/lstm_training_history.png')
plt.show()

# %%
# Make predictions on the test set
y_pred = model.predict(X_test)

# Inverse transform the predictions and actual values
y_test_inv = scaler_y.inverse_transform(y_test)
y_pred_inv = scaler_y.inverse_transform(y_pred)

# Calculate performance metrics
mse = mean_squared_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_inv, y_pred_inv)
r2 = r2_score(y_test_inv, y_pred_inv)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# %%
# Plot predictions vs actual values
plt.figure(figsize=(14, 7))

# Plot a subset of the test data for better visualization
sample_size = min(500, len(y_test_inv))
indices = np.arange(sample_size)

plt.plot(indices, y_test_inv[:sample_size], 'b-', label='Actual Power Output')
plt.plot(indices, y_pred_inv[:sample_size], 'r-', label='Predicted Power Output')
plt.title('LSTM Model: Actual vs Predicted PV Power Output')
plt.xlabel('Time Steps')
plt.ylabel('Power Output (kW)')
plt.legend()
plt.grid(True)

plt.savefig('results/lstm_predictions.png')
plt.show()

# %%
# Create a scatter plot of actual vs predicted values
plt.figure(figsize=(10, 8))
plt.scatter(y_test_inv, y_pred_inv, alpha=0.5)
plt.plot([y_test_inv.min(), y_test_inv.max()], [y_test_inv.min(), y_test_inv.max()], 'r--')
plt.title('LSTM Model: Actual vs Predicted Power Output')
plt.xlabel('Actual Power Output (kW)')
plt.ylabel('Predicted Power Output (kW)')
plt.grid(True)

plt.savefig('results/lstm_scatter.png')
plt.show()

# %% [markdown]
# ## Save the Final Model

# %%
# Save the model with the specific configuration name
model_save_path = f"models/lstm/station_config_{station_config_2['config_id']}_model.keras"
model.save(model_save_path)
print(f"Model saved successfully to {model_save_path}")

# Also save as the default model for backward compatibility
model.save('models/power_forecast_model.keras')
print("Model also saved as the default model for backward compatibility.")

# %% [markdown]
# ## Conclusion
#
# This notebook demonstrates the implementation of an LSTM model for PV power forecasting. The model uses historical weather data and power output to predict future power generation. The performance metrics and visualizations provide insights into the model's accuracy and limitations.
#
# Next steps could include:
# 1. Hyperparameter tuning to improve model performance
# 2. Feature engineering to incorporate additional relevant variables
# 3. Testing different model architectures (e.g., bidirectional LSTM, GRU)
# 4. Implementing ensemble methods for more robust predictions

# %%
# Class implementation for use with run_lstm_models.py
class LSTMForecaster:
    def __init__(self, sequence_length=24, batch_size=32, epochs=50, feature_set='station'):
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Configuration for station_config_2
        self.config = {
            'config_id': 2,
            'feature_set': feature_set,  # Use the provided feature_set parameter
            'lstm_units': [128, 64, 32],
            'dense_units': [32, 16],
            'dropout_rates': [0.3, 0.3, 0.3],
            'learning_rate': 0.001,
            'bidirectional': True,
            'batch_norm': True,
            'l1_reg': 0.0,
            'l2_reg': 0.001,
            'optimizer': 'adam'
        }
    
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
                'hour_cos',
                'day_cos',   # Added day_cos as requested
                'day_sin',   # Added day_sin as requested
                'isNight'    # Added isNight as requested
            ],
            'station': [
                'Station_GlobalRadiation [W m-2]',
                'Station_Temperature [degree_Celsius]',
                'Station_WindSpeed [m s-1]',
                'Station_ClearSkyIndex',
                'hour_sin',
                'hour_cos',
                'day_cos',   # Added day_cos as requested
                'day_sin',   # Added day_sin as requested
                'isNight'    # Added isNight as requested
            ],
            'combined': [
                'Combined_GlobalRadiation [W m-2]',
                'Combined_Temperature [degree_Celsius]',
                'Combined_WindSpeed [m s-1]',
                'Combined_ClearSkyIndex',
                'hour_sin',
                'hour_cos',
                'day_cos',   # Added day_cos as requested
                'day_sin',   # Added day_sin as requested
                'isNight'    # Added isNight as requested
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
    
    def run_pipeline(self, data_path):
        """Run the full LSTM forecasting pipeline"""
        print(f"Running LSTM forecasting pipeline with {self.config['feature_set']} configuration {self.config['config_id']}")
        
        # Load data
        df = pd.read_parquet(data_path)
        print(f"Loaded data shape: {df.shape}")
        
        # Get feature sets
        feature_sets = self.prepare_feature_sets(df)
        
        # Select the appropriate feature set based on configuration
        feature_set_name = self.config['feature_set']
        if feature_set_name not in feature_sets:
            print(f"Warning: Feature set '{feature_set_name}' not found. Using 'station' as default.")
            feature_set_name = 'station'
        
        # Get the features for the selected feature set
        features = feature_sets[feature_set_name]
        
        # Separate features that need scaling from those that don't
        time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'isNight']
        features_to_scale = [f for f in features if f not in time_features]
        additional_features = [f for f in features if f in time_features]
        
        print(f"Using feature set: {feature_set_name}")
        print(f"Features to scale: {features_to_scale}")
        print(f"Additional features: {additional_features}")
        
        target = 'power_w'
        
        # Data normalization
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        scaled_features = scaler_x.fit_transform(df[features_to_scale])
        unscaled_features = df[additional_features].values
        X = np.hstack((scaled_features, unscaled_features))
        y = scaler_y.fit_transform(df[[target]])
        
        # Save the scalers
        os.makedirs('models', exist_ok=True)
        joblib.dump(scaler_x, 'models/scaler_x.pkl')
        joblib.dump(scaler_y, 'models/scaler_y.pkl')
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X, y, self.sequence_length)
        print(f"Sequence shapes - X: {X_seq.shape}, y: {y_seq.shape}")
        
        # Split data
        train_size = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:train_size], X_seq[train_size:]
        y_train, y_test = y_seq[:train_size], y_seq[train_size:]
        
        # Build and train model
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = self._build_model(input_shape)
        
        # Define callbacks
        callbacks = self._create_callbacks()
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        metrics = self._evaluate_model(model, X_test, y_test, scaler_y)
        
        # Save model
        model_path = f"models/lstm/{self.config['feature_set']}_config_{self.config['config_id']}_model.keras"
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Plot results
        self._plot_results(history, y_test, model.predict(X_test), scaler_y)
        
        return {self.config['feature_set']: metrics}
    
    def _create_sequences(self, X, y, time_steps):
        """Create sequences for time series forecasting"""
        X_seq, y_seq = [], []
        for i in range(len(X) - time_steps):
            X_seq.append(X[i:i + time_steps])
            y_seq.append(y[i + time_steps])
        return np.array(X_seq), np.array(y_seq)
    
    def _build_model(self, input_shape):
        """Build the LSTM model with the specified configuration"""
        model = Sequential()
        
        # Extract configuration parameters
        lstm_units = self.config['lstm_units']
        dense_units = self.config['dense_units']
        dropout_rates = self.config['dropout_rates']
        learning_rate = self.config['learning_rate']
        bidirectional = self.config.get('bidirectional', False)
        batch_norm = self.config.get('batch_norm', False)
        l1_regularization = self.config.get('l1_reg', 0.0)
        l2_regularization = self.config.get('l2_reg', 0.0)
        
        # Regularizer
        regularizer = None
        if l1_regularization > 0 or l2_regularization > 0:
            regularizer = l1_l2(l1=l1_regularization, l2=l2_regularization)
        
        # First LSTM layer
        if bidirectional:
            model.add(Bidirectional(
                LSTM(units=lstm_units[0], return_sequences=True if len(lstm_units) > 1 else False,
                     kernel_regularizer=regularizer),
                input_shape=input_shape))
        else:
            model.add(LSTM(units=lstm_units[0], return_sequences=True if len(lstm_units) > 1 else False,
                           kernel_regularizer=regularizer, input_shape=input_shape))
        
        if batch_norm:
            model.add(BatchNormalization())
        
        model.add(Dropout(dropout_rates[0]))
        
        # Middle LSTM layers (if any)
        for i in range(1, len(lstm_units) - 1):
            if bidirectional:
                model.add(Bidirectional(LSTM(units=lstm_units[i], return_sequences=True,
                                             kernel_regularizer=regularizer)))
            else:
                model.add(LSTM(units=lstm_units[i], return_sequences=True,
                               kernel_regularizer=regularizer))
            
            if batch_norm:
                model.add(BatchNormalization())
            
            model.add(Dropout(dropout_rates[min(i, len(dropout_rates) - 1)]))
        
        # Last LSTM layer (if more than one)
        if len(lstm_units) > 1:
            if bidirectional:
                model.add(Bidirectional(LSTM(units=lstm_units[-1], return_sequences=False,
                                             kernel_regularizer=regularizer)))
            else:
                model.add(LSTM(units=lstm_units[-1], return_sequences=False,
                               kernel_regularizer=regularizer))
            
            if batch_norm:
                model.add(BatchNormalization())
            
            model.add(Dropout(dropout_rates[min(len(lstm_units) - 1, len(dropout_rates) - 1)]))
        
        # Dense layers
        for i, units in enumerate(dense_units):
            model.add(Dense(units, activation='relu', kernel_regularizer=regularizer))
            
            if batch_norm:
                model.add(BatchNormalization())
            
            if i < len(dense_units) - 1:  # No dropout after the last dense layer
                model.add(Dropout(dropout_rates[min(len(lstm_units) + i, len(dropout_rates) - 1)]))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile the model
        if self.config['optimizer'].lower() == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif self.config['optimizer'].lower() == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            optimizer = self.config['optimizer']
        
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def _create_callbacks(self):
        """Create callbacks for model training"""
        os.makedirs('models/lstm', exist_ok=True)
        os.makedirs(f'logs/{self.config["feature_set"]}_config_{self.config["config_id"]}', exist_ok=True)
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=f'models/lstm/{self.config["feature_set"]}_config_{self.config["config_id"]}_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=f'logs/{self.config["feature_set"]}_config_{self.config["config_id"]}',
            histogram_freq=1
        )
        
        return [early_stopping, model_checkpoint, tensorboard_callback]
    
    def _evaluate_model(self, model, X_test, y_test, scaler_y):
        """Evaluate the model and return metrics"""
        y_pred = model.predict(X_test)
        
        # Inverse transform
        y_test_inv = scaler_y.inverse_transform(y_test)
        y_pred_inv = scaler_y.inverse_transform(y_pred)
        
        # Calculate metrics
        mse = mean_squared_error(y_test_inv, y_pred_inv)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        r2 = r2_score(y_test_inv, y_pred_inv)
        
        print(f"\nModel Evaluation ({self.config['feature_set']} config {self.config['config_id']}):")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def _plot_results(self, history, y_test, y_pred, scaler_y):
        """Plot and save training history and prediction results"""
        os.makedirs('results', exist_ok=True)
        
        # Inverse transform
        y_test_inv = scaler_y.inverse_transform(y_test)
        y_pred_inv = scaler_y.inverse_transform(y_pred)
        
        # Plot training history
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{self.config["feature_set"].capitalize()} Config {self.config["config_id"]} - Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title(f'{self.config["feature_set"].capitalize()} Config {self.config["config_id"]} - Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'results/{self.config["feature_set"]}_config_{self.config["config_id"]}_history.png')
        
        # Plot predictions vs actual
        plt.figure(figsize=(14, 7))
        
        sample_size = min(500, len(y_test_inv))
        indices = np.arange(sample_size)
        
        plt.plot(indices, y_test_inv[:sample_size], 'b-', label='Actual Power Output')
        plt.plot(indices, y_pred_inv[:sample_size], 'r-', label='Predicted Power Output')
        plt.title(f'{self.config["feature_set"].capitalize()} Config {self.config["config_id"]} - Actual vs Predicted PV Power')
        plt.xlabel('Time Steps')
        plt.ylabel('Power Output (W)')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(f'results/{self.config["feature_set"]}_config_{self.config["config_id"]}_predictions.png')
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(y_test_inv, y_pred_inv, alpha=0.5)
        plt.plot([y_test_inv.min(), y_test_inv.max()], [y_test_inv.min(), y_test_inv.max()], 'r--')
        plt.title(f'{self.config["feature_set"].capitalize()} Config {self.config["config_id"]} - Actual vs Predicted')
        plt.xlabel('Actual Power Output (W)')
        plt.ylabel('Predicted Power Output (W)')
        plt.grid(True)
        
        plt.savefig(f'results/{self.config["feature_set"]}_config_{self.config["config_id"]}_scatter.png')