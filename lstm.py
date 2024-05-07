# -*- coding: utf-8 -*-
"""
Author: Michael Grün
Email: michaelgruen@hotmail.com
Description: This script calculates the power output of a photovoltaic system
             using the PVLib library and processes the results in an Excel file.
Version: 1.0
Date: 2024-05-03
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import roc_curve


def split_sequences(X, y, n_steps):
    """
    Split sequences into input (X) and output (y) sequences.

    Args:
        X (np.ndarray): Input sequences.
        y (np.ndarray): Output sequences.
        n_steps (int): Number of time steps in each sequence.

    Returns:
        tuple: Tuple containing:
            X_ (np.ndarray): Input sequences.
            y_ (np.ndarray): Output sequences.
    """
    X_, y_ = list(), list()
    for i in range(len(X)):
        end_ix = i + n_steps
        if end_ix > len(X):
            break
        seq_x, seq_y = X[i:end_ix], y[end_ix - 1]
        X_.append(seq_x)
        y_.append(seq_y)
    return np.array(X_), np.array(y_)


def build_lstm_model(n_steps, n_features):
    """
    Build and compile an LSTM model.

    Args:
        n_steps (int): Number of time steps in each sequence.
        n_features (int): Number of features in the input data.

    Returns:
        keras.models.Sequential: Compiled LSTM model.
    """
    model = Sequential([
        LSTM(100, input_shape=(n_steps, n_features), activation='relu', return_sequences=True),
        LSTM(100, activation='relu', dropout=0.2),
        LSTM(100, activation='relu'),
        Dense(1)
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['mse', 'mae'])
    return model


def plot_predictions(data, test_indices, predictions, y_test_unsc):
    """
    Plot the actual and predicted EV load.

    Args:
        data (pd.DataFrame): Original data frame.
        test_indices (list): List of test data indices.
        predictions (np.ndarray): Predicted EV load values.
        y_test_unsc (np.ndarray): Actual EV load values for the test set.
    """
    pred_df = data.iloc[test_indices][["Date", "LocalDt", "AC_Power"]]
    pred_df["y_hat"] = predictions
    pred_df["LocalDt"] = pd.to_datetime(pred_df["LocalDt"], format="%m/%d/%Y %H:%M")
    pred_df["RealTime"] = pred_df["LocalDt"] - pd.timedelta(hours=1)
    pred_df = pred_df.sort_values(["RealTime"], ascending=True)
    pred_df = pred_df.set_index(pd.to_datetime(pred_df.RealTime), drop=True)

    plt.figure(figsize=(12, 5))
    plt.plot(pred_df.index, pred_df["y_hat"], color='blue', label="1-Day-Ahead EV-Load Forecast (KW)")
    plt.plot(pred_df.index, pred_df["AC_Power"], color='red', label="Actual EV-Load (KW)")
    plt.ylabel('EV-Load (KW)', fontsize=12)
    plt.title('Actual EV Load vs. 1-Day-Ahead Forecast')
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()


def plot_roc_like_curve(y_test_unsc, predictions):
    """
    Plot the ROC-like curve for regression.

    Args:
        y_test_unsc (np.ndarray): Actual EV load values for the test set.
        predictions (np.ndarray): Predicted EV load values.
    """
    threshold_values = np.linspace(450, 550, 10)
    tpr_values, fpr_values = [], []

    for threshold in threshold_values:
        binary_predictions = (predictions > threshold).astype(int)
        fpr, tpr, _ = roc_curve(y_test_unsc > threshold, binary_predictions)
        tpr_values.append(tpr[[1]] if len(tpr) > 1 else np.nan)
        fpr_values.append(fpr[[1]] if len(fpr) > 1 else np.nan)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_values, tpr_values, color='blue', lw=2, label='ROC-like Curve')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random Guess')

    for i, threshold in enumerate(threshold_values):
        plt.scatter(fpr_values[i], tpr_values[i], color='red', marker='o', label=f'Threshold = {threshold:.2f}')
        plt.annotate(f'{threshold:.2f}', (fpr_values[i], tpr_values[i]), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC-like Curve for Regression with Thresholds')
    plt.legend()
    plt.show()


def save_results_to_excel(data, test_indices, predictions, y_test_unsc):
    """
    Save predictions and ROC-like curve results to Excel files.

    Args:
        data (pd.DataFrame): Original data frame.
        test_indices (list): List of test data indices.
        predictions (np.ndarray): Predicted EV load values.
        y_test_unsc (np.ndarray): Actual EV load values for the test set.
    """
    pred_df = data.iloc[test_indices][["Date", "LocalDt", "AC_Power"]]
    pred_df["y_hat"] = predictions
    pred_df.to_excel('predictions_values.xlsx', index=False)

    threshold_values = np.linspace(450, 550, 10)
    tpr_values, fpr_values = [], []

    for threshold in threshold_values:
        binary_predictions = (predictions > threshold).astype(int)
        fpr, tpr, _ = roc_curve(y_test_unsc > threshold, binary_predictions)
        tpr_values.append(tpr[[1]] if len(tpr) > 1 else np.nan)
        fpr_values.append(fpr[[1]] if len(fpr) > 1 else np.nan)

    roc_results_data = {
        'Threshold': threshold_values,
        'True Positive Rate (Sensitivity)': tpr_values,
        'False Positive Rate': fpr_values
    }
    roc_results_df = pd.DataFrame(roc_results_data)
    roc_results_df.to_excel('roc_results.xlsx', index=False)


def main():
    """
    Main function to run the EV load prediction and analysis.
    """
    # Load data
    data = pd.read_excel("PV.xlsx")

    # Preprocess data
    test_dates = pd.to_datetime(['2020-12-23', '2020-12-24', '2020-12-25', '2020-12-26', '2020-12-27', '2020-12-28', '2020-12-29'])
    test_df = data[data['Date'].isin(test_dates)]
    test_indices = list(test_df.index)
    steps_minus1 = 8
    test_i = [x - steps_minus1 for x in test_indices]
    train_i = [x for x in list(data.index) if x not in test_i]
    train_i = train_i[:-steps_minus1]

    # Scale data
    qual_vars = ['Cell Temperature', 'temp_air', 'wind_speed']
    quan_vars = ['poa_diffuse', 'poa_direct', 'cos_D', 'sin_D', 'poa_global', 'sin_H', 'cos_M', 'sin_M', 'cos_H']
    sc_x = StandardScaler().fit(data[qual_vars])
    sc_y = StandardScaler().fit(data["AC_Power"].values.reshape(-1, 1))
    data[qual_vars] = sc_x.transform(data[qual_vars])
    data["AC_Power"] = sc_y.transform(data["AC_Power"].values.reshape(-1, 1))
    data = data[qual_vars + quan_vars + ['AC_Power']]

    # Prepare data for LSTM
    features = data.columns[:-1]
    X = data[features].values
    y = data['AC_Power'].values
    n_steps = steps_minus1 + 1
    X, y = split_sequences(X, y, n_steps)
    x_train, y_train = X[train_i], y[train_i]
    x_test, y_test = X[test_i], y[test_i]

    # Build LSTM model
    model = build_lstm_model(x_train.shape[1], x_train.shape[2])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=[early_stopping], verbose=1)

    # Evaluate model
    predictions = sc_y.inverse_transform(model.predict(x_test).reshape(-1, 1))
    y_test_unsc = sc_y.inverse_transform(y_test.reshape(-1, 1))
    print(f"MSE: {mean_squared_error(y_test_unsc, predictions):.4f}")
    print(f"MAE: {mean_absolute_error(y_test_unsc, predictions):.4f}")

    # Plot results
    plot_predictions(data, test_indices, predictions, y_test_unsc)

    # Calculate ROC-like curve
    plot_roc_like_curve(y_test_unsc, predictions)

    # Save results to Excel
    save_results_to_excel(data, test_indices, predictions, y_test_unsc)


if __name__ == "__main__":
    sys.exit(main())

