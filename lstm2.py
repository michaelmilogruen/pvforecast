
import pandas as pd
import numpy as np
import itertools
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

# # Make predictions on the test set
# yhat = model.predict(x_test, verbose=0)
# predictions = sc_y.inverse_transform(yhat)
# y_test_unsc = sc_y.inverse_transform(y_test.reshape(-1, 1))
#
# threshold_values = np.linspace(450, 550, 10)  # Adjust the range according to your needs
# tpr_values = []
# fpr_values = []
#
# # Calculate True Positive Rate (Sensitivity) and False Positive Rate for each threshold
# for threshold in threshold_values:
#     # Convert continuous predictions to binary based on the threshold
#     binary_predictions = (predictions > threshold).astype(int)
#
#     # Calculate True Positive Rate and False Positive Rate
#     fpr, tpr, _ = roc_curve(y_test_unsc > threshold, binary_predictions)
#
#     # Append values to the lists
#     tpr_values.append(tpr[1])  # Use tpr[1] since it corresponds to sensitivity
#     fpr_values.append(fpr[1])
#
# # Plot the ROC-like curve
# plt.figure(figsize=(8, 6))
# plt.plot(fpr_values, tpr_values, color='blue', lw=2, label='ROC-like Curve')
# plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random Guess')
#
# plt.xlabel('False Positive Rate (FPR)')
# plt.ylabel('True Positive Rate (TPR)')
# plt.title('ROC-like Curve for Regression')
# plt.legend()
# plt.show()
#
#
# tn_values = []
# fn_values = []
# binary_true_values = y_test_unsc > threshold
#
# # Calculate True Negative Rate (Specificity) and False Negative Rate for each threshold
# for threshold in threshold_values:
#     # Convert continuous predictions to binary based on the threshold
#     binary_predictions = (predictions > threshold).astype(int)
#
#     # Calculate True Positives, True Negatives, False Positives, False Negatives
#     tp = np.sum((binary_true_values == 1) & (binary_predictions == 1))
#     tn = np.sum((binary_true_values == 0) & (binary_predictions == 0))
#     fp = np.sum((binary_true_values == 0) & (binary_predictions == 1))
#     fn = np.sum((binary_true_values == 1) & (binary_predictions == 0))
#
#     # Calculate True Negative Rate (Specificity) and False Negative Rate
#     specificity = tn / (tn + fp)
#     sensitivity = tp / (tp + fn)
#
#     # Append values to the lists
#     tn_values.append(specificity)
#     fn_values.append(1 - sensitivity)  # False Negative Rate is complementary to Sensitivity
#
# # Plot the ROC-like curve including Specificity and False Negative Rate
# plt.figure(figsize=(8, 6))
# plt.plot(fpr_values, tpr_values, color='blue', lw=2, label='ROC-like Curve')
# plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random Guess')
# plt.plot(fpr_values, tn_values, color='green', lw=2, linestyle='--', label='Specificity')
# plt.plot(fpr_values, fn_values, color='red', lw=2, linestyle='--', label='False Negative Rate')
#
# plt.xlabel('False Positive Rate (FPR)')
# plt.ylabel('Rates')
# plt.title('ROC-like Curve for Regression with Specificity and False Negative Rate')
# plt.legend()
# plt.show()
#
# # Plot the ROC-like curve
# plt.figure(figsize=(8, 6))
# plt.plot(fpr_values, tpr_values, color='blue', lw=2, label='ROC-like Curve')
# plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random Guess')
# # Calculate True Positive Rate (Sensitivity) and False Positive Rate for each threshold
# for threshold in threshold_values:
#     # Convert continuous predictions to binary based on the threshold
#     binary_predictions = (predictions > threshold).astype(int)
#
#     # Calculate True Positive Rate and False Positive Rate
#     fpr, tpr, _ = roc_curve(y_test_unsc > threshold, binary_predictions)
#
#     # Append values to the lists
#     tpr_values.append(tpr[1] if len(tpr) > 1 else np.nan)
#     fpr_values.append(fpr[1] if len(fpr) > 1 else np.nan)
#
# # Check the lengths of the arrays after the loop
# print(f"Length of Threshold Values: {len(threshold_values)}")
# print(f"Length of TPR Values: {len(tpr_values)}")
# print(f"Length of FPR Values: {len(fpr_values)}")
#
#
# # Plot the threshold points on the ROC-like curve
# for i, threshold in enumerate(threshold_values):
#     plt.scatter(fpr_values[i], tpr_values[i], color='red', marker='o', label=f'Threshold = {threshold:.2f}')
#     plt.annotate(f'{threshold:.2f}', (fpr_values[i], tpr_values[i]), textcoords="offset points", xytext=(0, 10), ha='center')
#
# plt.xlabel('False Positive Rate (FPR)')
# plt.ylabel('True Positive Rate (TPR)')
# plt.title('ROC-like Curve for Regression with Thresholds')
# plt.legend()
# plt.show()
#
# selected_threshold = 550 # Adjust the threshold as needed
#
# # Convert continuous predictions to binary based on the selected threshold
# binary_predictions = (predictions > selected_threshold).astype(int)
#
# # Plot Frequency vs. Test Results
# plt.figure(figsize=(8, 6))
# plt.hist(predictions[binary_predictions == 1], bins=50, color='blue', alpha=0.7, label='Positive Class (Predicted)')
# plt.hist(predictions[binary_predictions == 0], bins=50, color='red', alpha=0.7, label='Negative Class (Predicted)')
# plt.axvline(x=selected_threshold, color='green', linestyle='--', label=f'Selected Threshold = {selected_threshold}')
#
# plt.xlabel('Test Results')
# plt.ylabel('Frequency')
# plt.title('Frequency vs. Test Results')
# plt.legend()
# plt.show()
#
# print("Length of Threshold Values:", len(threshold_values))
# print("Length of TPR Values:", len(tpr_values))
# print("Length of FPR Values:", len(fpr_values))
#
#
# predictions_data = {
#     'DateTime': chart_data.index,
#     'Predicted_EV_Load_KW': chart_data["y_hat"],
#     'Actual_EV_Load_KW': chart_data["AC_Power"]
# }
#
# # Create a DataFrame
# predictions_df = pd.DataFrame(predictions_data)
#
# # Create an Excel writer
# with pd.ExcelWriter('predictions_values.xlsx') as writer:
#     # Save predictions DataFrame
#     predictions_df.to_excel(writer, sheet_name='Predictions_Values', index=False)
#
# if len(threshold_values) == len(tpr_values) == len(fpr_values):
#     roc_results_data = {
#         'Threshold': threshold_values,
#         'True Positive Rate (Sensitivity)': tpr_values,
#         'False Positive Rate': fpr_values
#     }
#
#     roc_results_df = pd.DataFrame(roc_results_data)
#
#     # Create an Excel writer for ROC results
#     with pd.ExcelWriter('roc_results.xlsx') as writer:
#         # Save ROC results DataFrame
#         roc_results_df.to_excel(writer, sheet_name='ROC_Results', index=False)
# else:
#     print("Error: Arrays must be of the same length.")
#
# roc_results_data = {
#     'Threshold': threshold_values,
#     'True Positive Rate (Sensitivity)': tpr_values,
#     'False Positive Rate': fpr_values
# }
#
# roc_results_df = pd.DataFrame(roc_results_data)
#
# with pd.ExcelWriter('roc_results.xlsx') as writer:
#     # Save ROC results DataFrame
#     roc_results_df.to_excel(writer, sheet_name='ROC_Results', index=False)
