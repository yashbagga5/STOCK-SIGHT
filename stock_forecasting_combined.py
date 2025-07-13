# stock_forecasting_combined.py
"""
Combined ARIMA, SARIMA, and LSTM forecasting for Apple (AAPL) stock data.
Ensure you have the following packages installed:
- pandas
- numpy
- matplotlib
- statsmodels
- scikit-learn
- tensorflow
- yfinance (for downloading data, optional)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings
warnings.filterwarnings("ignore")

# 1. Load and clean data
# If you already have 'AAPL_stock_data_cleaned.csv', use it. Otherwise, use the raw CSV and clean it.
try:
    df = pd.read_csv("AAPL_stock_data_cleaned.csv", parse_dates=['Date'], index_col='Date')
    df.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
except Exception:
    df = pd.read_csv("AAPL_stock_data.csv", skiprows=2)
    df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df[['Close', 'High', 'Low', 'Open', 'Volume']]
    df.to_csv("AAPL_stock_data_cleaned.csv")

# Keep only 'Close' for univariate forecasting
close_data = df['Close'].astype(float)

# 2. ARIMA Forecasting
print("\n=== ARIMA Forecasting ===")
train_arima = close_data[:-30]
test_arima = close_data[-30:]
model_arima = ARIMA(train_arima, order=(5,1,0))
model_arima_fit = model_arima.fit()
arima_forecast = model_arima_fit.forecast(steps=30)
arima_mse = mean_squared_error(test_arima, arima_forecast)
print(f"ARIMA Test MSE: {arima_mse:.2f}")

# 3. SARIMA (SARIMAX) Forecasting
print("\n=== SARIMA (SARIMAX) Forecasting ===")
train_sarima = close_data[:-30]
test_sarima = close_data[-30:]
model_sarima = SARIMAX(train_sarima, order=(1,1,1), seasonal_order=(1,1,1,12))
model_sarima_fit = model_sarima.fit(disp=False)
sarima_forecast = model_sarima_fit.predict(start=len(train_sarima), end=len(train_sarima)+len(test_sarima)-1)
sarima_mse = mean_squared_error(test_sarima, sarima_forecast)
print(f"SARIMA Test MSE: {sarima_mse:.2f}")

# 4. LSTM Forecasting
print("\n=== LSTM Forecasting ===")
# Normalize data
df_lstm = close_data.to_frame()
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_lstm)

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

sequence_length = 60
X, y = create_sequences(scaled_data, sequence_length)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM model
model_lstm = Sequential()
model_lstm.add(LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)))
model_lstm.add(LSTM(50))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mean_squared_error')
model_lstm.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Predict
predicted = model_lstm.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
lstm_mse = mean_squared_error(actual_prices, predicted_prices)
print(f"LSTM Test MSE: {lstm_mse:.2f}")

# 5. Plotting
plt.figure(figsize=(14, 6))
plt.plot(test_arima.index, test_arima.values, label='Actual (Last 30)', color='black')
plt.plot(test_arima.index, arima_forecast, label='ARIMA Forecast', linestyle='--', color='blue')
plt.plot(test_sarima.index, sarima_forecast, label='SARIMA Forecast', linestyle=':', color='green')
plt.title('ARIMA & SARIMA Forecast vs Actual')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(actual_prices, label='Actual (LSTM)', color='black')
plt.plot(predicted_prices, label='LSTM Predicted', color='orange')
plt.title('LSTM Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# 6. Comparison Section: Combined Plot
plt.figure(figsize=(14, 6))
plt.plot(test_arima.index, test_arima.values, label='Actual (Last 30)', color='black')
plt.plot(test_arima.index, arima_forecast, label='ARIMA Forecast', linestyle='--', color='blue')
plt.plot(test_sarima.index, sarima_forecast, label='SARIMA Forecast', linestyle=':', color='green')
# For LSTM, align the last 30 predictions with the test set
if len(predicted_prices) >= 30:
    plt.plot(test_arima.index, predicted_prices[-30:], label='LSTM Forecast', linestyle='-.', color='orange')
plt.title('Model Comparison: Forecasts vs Actual (Last 30 Days)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# 7. Summary Table
summary = pd.DataFrame({
    'Model': ['ARIMA', 'SARIMA', 'LSTM'],
    'MSE': [arima_mse, sarima_mse, lstm_mse]
})
print("\n=== Model Performance Summary ===")
print(summary)
