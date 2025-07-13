# Stock_Sight
# 📈 Stock Sight: AAPL Stock Forecasting with ARIMA & SARIMA

This project forecasts the stock prices of **Apple Inc. (AAPL)** using two popular time series models:
- 🔁 **ARIMA**
- 🔄 **SARIMA**

It compares the performance of these models and visualizes how each predicts future stock prices based on historical data.

---

## 📂 Project Structure

-  ├── AAPL_stock_data.csv # Raw stock data (from Yahoo Finance or similar)
-  ├── stock_forecasting_combined.py # Main Python script for forecasting
-  ├── AAPL_stock_data_cleaned.csv # Cleaned stock data 
-  ├── README.md # Project documentation


---

## 🚀 Getting Started

 1. Install Required Libraries


pip install pandas numpy matplotlib statsmodels scikit-learn tensorflow yfinance

2. Run


python stock_forecasting_combined.py

---



The script will:
- Clean the raw stock data
- Train three models: ARIMA, SARIMA
- Forecast future prices
- Display plots for comparison
- Print MSE (Mean Squared Error) for each model

---
🔧 Models Used


Model:Description
- ARIMA:	AutoRegressive Integrated Moving Average
- SARIMA:	Seasonal ARIMA with trend + seasonality

Each model predicts the next 30 days of stock prices and compares with actuals.

--- 

📊 Output
- Visual comparison of predicted vs actual prices
- Performance table with Mean Squared Errors
- Insights into model accuracy and behavior

---

📌 Notes
- Input data: Daily AAPL stock prices with columns Date, Close, High, Low, Open, Volume
- Cleaned data is cached to AAPL_stock_data_cleaned.csv


---
=== Model Performance Summary ===
-    Model     MSE
- 1  ARIMA     3.45
- 2 SARIMA     2.81

---

🧠 Future Improvements
- Hyperparameter tuning
- Incorporating additional features like volume or news sentiment
- Real-time prediction API

---
🤝 Contributions


Open to collaboration and enhancements! Feel free to fork and improve the project.

---
📬 Contact


Made by Yash Bagga – Data Science Enthusiast
| Mail: yashbagga5@email.com
