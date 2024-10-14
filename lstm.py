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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_curve
from tensorflow.keras.layers import Input

def split_sequences(X, y, n_steps):
    X_, y_ = list(), list()
    for i in range(len(X) - n_steps + 1):
        seq_x, seq_y = X[i:i + n_steps], y[i + n_steps - 1]
        X_.append(seq_x)
        y_.append(seq_y)
    return np.array(X_), np.array(y_)


def build_lstm_model(n_steps, n_features):
    model = Sequential([
        Input(shape=(n_steps, n_features)),
        Bidirectional(LSTM(120, activation='relu', return_sequences=True)),
        LSTM(120, activation='relu', dropout=0.3, return_sequences=True),
        LSTM(120, activation='relu'),
        Dense(1)
    ])
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.0005), metrics=['mse', 'mae'])
    return model


def plot_predictions(data, test_indices, predictions, y_test_unsc):
    pred_df = data.iloc[test_indices[:len(predictions)]][["timestamp", "AC_Power"]].copy()
    pred_df.loc[:, "y_hat"] = predictions
    pred_df["LocalDt"] = pd.to_datetime(pred_df["timestamp"], format="%Y-%m-%d %H:%M:%S")
    pred_df["RealTime"] = pred_df["LocalDt"] - pd.Timedelta(hours=1)
    pred_df = pred_df.sort_values(["RealTime"], ascending=True)
    pred_df = pred_df.set_index(pd.to_datetime(pred_df.RealTime), drop=True)

    plt.figure(figsize=(12, 5))
    plt.plot(pred_df.index, pred_df["y_hat"], color='blue', label="1-Day-Ahead PV Power Forecast (KW)")
    plt.plot(pred_df.index, pred_df["AC_Power"], color='red', label="Actual PV Power (KW)")
    plt.ylabel('PV Power (KW)', fontsize=12)
    plt.title('Actual PV Power vs. 1-Day-Ahead Forecast')
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_roc_curve(y_test_unsc, predictions):
    threshold_values = np.linspace(min(y_test_unsc), max(y_test_unsc), 10)
    tpr_values, fpr_values = [], []

    for threshold in threshold_values:
        binary_predictions = (predictions > threshold).astype(int)
        fpr, tpr, _ = roc_curve(y_test_unsc > threshold, binary_predictions)
        tpr_values.append(tpr[1] if len(tpr) > 1 else np.nan)
        fpr_values.append(fpr[1] if len(fpr) > 1 else np.nan)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_values, tpr_values, color='blue', lw=2, label='ROC-like Curve')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random Guess')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC-like Curve for Regression')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    data = pd.read_excel("results.xlsx", sheet_name='Model Chain Results', usecols='A,B,L,M,P')
    data.columns = ['timestamp', 'AC_Power', 'temp_air', 'wind_speed', 'global_irradiation']
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['hour'] = data['timestamp'].dt.hour
    data['sin_hour'] = np.sin(2 * np.pi * data['hour'] / 24.0)
    data['cos_hour'] = np.cos(2 * np.pi * data['hour'] / 24.0)
    data['month'] = data['timestamp'].dt.month
    data['sin_month'] = np.sin(2 * np.pi * data['month'] / 12.0)
    data['cos_month'] = np.cos(2 * np.pi * data['month'] / 12.0)

    features = ['sin_hour', 'cos_hour', 'sin_month', 'cos_month', 'temp_air', 'wind_speed', 'global_irradiation']
    target = 'AC_Power'

    train_size = int(len(data) * 0.8)
    train_df = data.iloc[:train_size].copy()
    test_df = data.iloc[train_size:].copy()

    sc_x = MinMaxScaler().fit(train_df[features])
    sc_y = MinMaxScaler().fit(train_df[target].values.reshape(-1, 1))
    train_df[features] = sc_x.transform(train_df[features])
    train_df[target] = sc_y.transform(train_df[target].values.reshape(-1, 1))
    test_df[features] = sc_x.transform(test_df[features])
    test_df[target] = sc_y.transform(test_df[target].values.reshape(-1, 1))

    X = train_df[features].values
    y = train_df[target].values
    n_steps = 24
    X, y = split_sequences(X, y, n_steps)

    x_train, y_train = X, y
    X_test = test_df[features].values
    y_test = test_df[target].values
    X_test, y_test = split_sequences(X_test, y_test, n_steps)
    x_test, y_test = X_test, y_test

    model = build_lstm_model(x_train.shape[1], x_train.shape[2])
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, mode='min')
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, callbacks=[checkpoint, EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)], verbose=1)

    model.load_weights('best_model.keras')

    predictions = sc_y.inverse_transform(model.predict(x_test).reshape(-1, 1))
    y_test_unsc = sc_y.inverse_transform(y_test.reshape(-1, 1))
    print(f"MSE: {mean_squared_error(y_test_unsc, predictions):.4f}")
    print(f"MAE: {mean_absolute_error(y_test_unsc, predictions):.4f}")
    mape = np.mean(np.abs((y_test_unsc - predictions) / y_test_unsc)) * 100
    print(f"MAPE: {mape:.4f}%")
    print(f"Average value of y_test_unsc: {np.mean(y_test_unsc):.4f}")
    print(f"Number of epochs made: {len(history.epoch)}")

    plot_predictions(data, test_df.index, predictions, y_test_unsc)
    plot_roc_curve(y_test_unsc, predictions)


if __name__ == "__main__":
    sys.exit(main())
