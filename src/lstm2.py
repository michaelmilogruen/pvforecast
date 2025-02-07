# -*- coding: utf-8 -*-
"""
Author: Michael Grün
Email: michaelgruen@hotmail.com
Description: This script calculates the power output of a photovoltaic system
             using the PVLib library and processes the results in an Excel file.
Version: 1.0
Date: 2024-05-03
"""

import pandas as pd
import numpy as np
import itertools
import os
# import datetime as dt
# from datetime import timedelta
# import pydot
# import pydotplus
# import gbraphviz
# import matplotlib.pyplot as plt
# from matplotli.pyplot import figure
# import sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
# import tensorflow as tf
# import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

Leoben = pd.read_excel(r"PV.xlsx")

steps_minus1 = 8

test_dates = ['2020-12-23','2020-12-24', '2020-12-25','2020-12-26','2020-12-27','2020-12-28','2020-12-29']

test_df = Leoben[Leoben['Date'].isin(test_dates)]

# Get indices associated with these observations
test_indices = list(test_df.index)

# Correct for the lag due to our stacking of time obvs
test_i = [x - steps_minus1 for x in test_indices]

# Get training indices
train_i = [x for x in list(Leoben.index) if x not in test_i]

# Correct for the lag due to our stacking of time obvs
train_i = train_i[: len(train_i) - steps_minus1]
con_scaled = Leoben.copy()

# Initialize scaler (here we use Min/Max)
sc = MinMaxScaler()

qual_vars = ['Cell Temperature', 'temp_air', 'wind_speed']

quan_vars = ['poa_diffuse','poa_direct','cos_D', 'sin_D', 'poa_global', 'sin_H', 'cos_M', 'sin_M', 'cos_H']

sc_x = StandardScaler().fit(con_scaled[qual_vars])
sc_y = StandardScaler().fit(con_scaled["AC_Power"].values.reshape(-1, 1))

# Scale quantitative variables
con_scaled[qual_vars] = sc_x.transform(con_scaled[qual_vars])
con_scaled["AC_Power"] = sc_y.transform(con_scaled["AC_Power"].values.reshape(-1, 1))

con_scaled = con_scaled[list(itertools.chain(qual_vars, quan_vars, ['AC_Power']))]
con_scaled.head()


in_seq1 = np.array(con_scaled["wind_speed"])
in_seq2 = np.array(con_scaled["poa_global"])
in_seq3 = np.array(con_scaled["sin_H"])
in_seq4 = np.array(con_scaled["sin_D"])
in_seq5 = np.array(con_scaled["cos_M"])
in_seq6 = np.array(con_scaled["sin_M"])
in_seq7 = np.array(con_scaled["temp_air"])
in_seq8 = np.array(con_scaled["cos_H"])
in_seq9 = np.array(con_scaled["cos_D"])
in_seq10 = np.array(con_scaled["poa_direct"])
in_seq11 = np.array(con_scaled["Cell Temperature"])
in_seq12 = np.array(con_scaled["poa_diffuse"])
out_seq = np.array(con_scaled["AC_Power"])

in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
in_seq3 = in_seq3.reshape((len(in_seq3), 1))
in_seq4 = in_seq4.reshape((len(in_seq4), 1))
in_seq5 = in_seq5.reshape((len(in_seq5), 1))
in_seq6 = in_seq6.reshape((len(in_seq6), 1))
in_seq7 = in_seq7.reshape((len(in_seq7), 1))
in_seq8 = in_seq8.reshape((len(in_seq8), 1))
in_seq9 = in_seq9.reshape((len(in_seq9), 1))
in_seq10 = in_seq10.reshape((len(in_seq10), 1))
in_seq11 = in_seq11.reshape((len(in_seq11), 1))
in_seq12 = in_seq12.reshape((len(in_seq12), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns

dataset = np.hstack((in_seq2,in_seq3,in_seq4, in_seq5,in_seq6, in_seq7,  in_seq8, in_seq9,in_seq10, in_seq11, in_seq12,out_seq))

def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

n_steps = steps_minus1 + 1
X, y = split_sequences(dataset, n_steps)

x_train = X[train_i]
x_test = X[test_i]
y_train = y[train_i]
y_test = y[test_i]

print(X.shape, y.shape)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

print("Input:")
print(*np.round(x_train[1],10), sep='\n')
print("\nOutput:")
print(np.round(y_train[1],4))

n_features = x_train.shape[2]

layer1 = LSTM(100, input_shape=(n_steps, n_features), activation='relu', return_sequences=True)
layer2 = LSTM(100,activation='relu', dropout=0.2)
layer3 = LSTM(100, activation='relu')
output = Dense(1)
layers_lstm = [layer1,layer2,output]

model = Sequential()
for layer in layers_lstm:
  model.add(layer)

# Define hyperparameters
loss = 'mse'
op = Adam(learning_rate=0.001)
metrics = ['mse','mae']
size = 10
n_epochs = 7
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Compile the model with the EarlyStopping callback
model.compile(loss=loss, optimizer=op, metrics=metrics)

# Fit the model with the EarlyStopping callback
history_lstm = model.fit(x_train, y_train,
                         validation_data=(x_test, y_test),
                         batch_size=size, epochs=n_epochs,
                         callbacks=[early_stopping], verbose=1)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(np.arange(1, len(history_lstm.history['loss']) + 1), history_lstm.history['loss'], color='olivedrab')
plt.title('Training set loss', fontdict={'fontsize': 16})
plt.xlabel('Epoch', fontdict={'fontsize': 14})
plt.ylabel('Loss', fontdict={'fontsize': 14})

plt.subplot(1, 2, 2)
plt.plot(np.arange(1, len(history_lstm.history['val_loss']) + 1), history_lstm.history['val_loss'], color='olivedrab')
plt.title('Test set loss', fontdict={'fontsize': 16})
plt.xlabel('Epoch', fontdict={'fontsize': 14})
plt.ylabel('Loss', fontdict={'fontsize': 14})

print(f"Training loss on the final epoch was: {history_lstm.history['loss'][-1]:0.4f}")

model.evaluate(x_test, y_test)

# Looking at the predictions
yhat = model.predict(x_test, verbose=0)
predictions = sc_y.inverse_transform(yhat)
y_test_unsc = sc_y.inverse_transform(y_test.reshape(-1,1))

print(f"MSE on unscaled data is: {mean_squared_error(y_test_unsc,predictions)}")
print(f"MAE on unscaled data is: {mean_absolute_error(y_test_unsc,predictions)}")

model.summary()

pred_df = Leoben[["Date","LocalDt","AC_Power"]].iloc[test_i] # Get dates with same indices as in our test dataset
pred_df["y_hat"] = predictions
pred_df.head()

pred_df = Leoben[["Date","LocalDt","AC_Power"]].iloc[test_indices] # Get dates with same indices as in our test dataset
pred_df["y_hat"] = predictions

firstweek = pred_df[pred_df["Date"].isin(test_dates)]
firstweek

chart_data = firstweek
chart_data.dtypes

chart_data["LocalDt"] = pd.to_datetime(chart_data["LocalDt"], format="%m/%d/%Y %H:%M")
chart_data["RealTime"] = chart_data["LocalDt"] - pd.to_timedelta(1, unit='h')

chart_data = chart_data.sort_values(["RealTime"], ascending=True)

chart_data = chart_data.set_index(pd.to_datetime(chart_data.RealTime), drop=True)
plt.style.use("default")
plt.figure(figsize=(12,5))
plt.xticks(fontsize=24)

ax1 = chart_data.y_hat.plot(color='blue', grid=False, label="1-Day-Ahead EV-Load Forecast (KW)")
ax1.set_ylim()
plt.yticks(fontsize=12)
ax2 = chart_data.AC_Power.plot(color='red', grid=False, secondary_y=True, label="Actual EV-Load (KW)")
ax2.set_ylim()
plt.yticks(fontsize=12)
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
plt.title('Actual EV Load vs. 1-Day-Ahead Forecast')
plt.legend(h1+h2, l1+l2, loc=2)
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(chart_data.index, chart_data["y_hat"], color='blue', label="1-Day-Ahead EV-Load Forecast (KW)")
plt.plot(chart_data.index, chart_data["AC_Power"], color='red', label="Actual EV-Load (KW)")


plt.ylabel('EV-Load (KW)', fontsize=12)
plt.title('Actual EV Load vs. 1-Day-Ahead Forecast')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.legend()
plt.show()
mse_percentage = (mean_squared_error(y_test_unsc, predictions) / np.mean(y_test_unsc)) * 100
mae_percentage = (mean_absolute_error(y_test_unsc, predictions) / np.mean(y_test_unsc)) * 100

print(f"MSE on unscaled data is: {mean_squared_error(y_test_unsc, predictions):.4f} ({mse_percentage:.2f}%)")
print(f"MAE on unscaled data is: {mean_absolute_error(y_test_unsc, predictions):.4f} ({mae_percentage:.2f}%)")
# Calculate the predictions for the last day of testing

rmse = np.sqrt(mean_squared_error(y_test_unsc, predictions))
print(f"RMSE on unscaled data is: {rmse:.4f}")

def calculate_mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

mape = calculate_mape(y_test_unsc, predictions)
print(f"MAPE on unscaled data is: {mape:.4f}%")

target_range = np.max(y_test_unsc) - np.min(y_test_unsc)

rmse_percentage = (rmse / np.mean(y_test_unsc)) * 100
print(f"RMSE as a percentage of the mean of actual values on unscaled data is: {rmse_percentage:.4f}%")

def calculate_smape(actual, predicted):
    return 200 * np.mean(np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted)))

# Make predictions on the test set and calculate SMAPE on unscaled data
smape = calculate_smape(y_test_unsc, predictions)
print(f"SMAPE on unscaled data is: {smape:.4f}%")
