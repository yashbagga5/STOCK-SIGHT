import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

# --- Streamlit Page Config ---
st.set_page_config(page_title="StockSight Dashboard", page_icon="📈", layout="wide")

# --- Custom CSS for dashboard style ---
st.markdown("""
    <style>
    .main {background-color: #f6f9fc;}
    .stButton>button {background-color: #4F8BF9; color: white; border-radius: 8px;}
    .stDataFrame {background-color: #f0f6ff; border-radius: 8px;}
    .section {
        background: #fff;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(79,139,249,0.07);
        padding: 2em 2em 1em 2em;
        margin-bottom: 2em;
    }
    .footer {text-align: center; color: #888; margin-top: 2em; font-size: 1.1em;}
    .sidebar-title {color: #4F8BF9; font-weight: bold; font-size: 1.5em;}
    .sidebar-about {font-size: 1em; color: #444;}
    .kpi-card {background: linear-gradient(90deg, #4F8BF9 60%, #A14FF9 100%); color: white; border-radius: 10px; padding: 1em; text-align: center; margin-bottom: 1em;}
    .kpi-label {font-size: 1.1em;}
    .kpi-value {font-size: 1.7em; font-weight: bold;}
    .highlight {background: #e6f7ff; border-radius: 8px; padding: 0.5em 1em;}
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.image("https://img.icons8.com/fluency/96/stock-share.png", width=80)
st.sidebar.markdown("<div class='sidebar-title'>StockSight</div>", unsafe_allow_html=True)
st.sidebar.markdown("""
**Instructions:**
- Upload your stock CSV (or use the default Apple data).
- The CSV must have columns: `Date, Close, High, Low, Open, Volume`.
- Explore the dashboard tabs for insights and forecasting.
""")
with st.sidebar.expander("About this app", expanded=False):
    st.markdown("""
    <div class='sidebar-about'>
    <b>StockSight</b> is a modern dashboard for stock price forecasting. Compare ARIMA, SARIMA, and Prophet models, view KPIs, and download results. Built with ❤️ using Streamlit.
    </div>
    """, unsafe_allow_html=True)

model_icons = {
    "ARIMA": "🔵 ARIMA",
    "SARIMA": "🟣 SARIMA",
    "Prophet": "🟠 Prophet",
    "Compare All": "✨ Compare All"
}

# --- File Upload and Data Cleaning ---
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, header=2)
    st.sidebar.success("Custom CSV uploaded successfully!")
else:
    df = pd.read_csv("AAPL_stock_data_cleaned.csv", header=2)
    st.sidebar.info("Using default Apple stock data.")

expected_cols = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
if list(df.columns)[:6] != expected_cols:
    df.columns = expected_cols
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
df = df[pd.to_numeric(df['Close'], errors='coerce').notnull()]
df['Close'] = df['Close'].astype(float)

# --- Data for modeling ---
data = df['Close']
train = data[:-30]
test = data[-30:]

# --- Helper: Moving Averages and Volatility ---
df['MA_50'] = df['Close'].rolling(window=50).mean()
df['MA_200'] = df['Close'].rolling(window=200).mean()
df['Volatility'] = df['Close'].rolling(window=30).std()

# --- Model Functions ---
def run_arima(train, test):
    from statsmodels.tsa.arima.model import ARIMA
    model = ARIMA(train, order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test))
    mse = mean_squared_error(test, forecast)
    return forecast, mse

def run_sarima(train, test):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
    model_fit = model.fit(disp=False)
    forecast = model_fit.predict(start=len(train), end=len(train)+len(test)-1)
    mse = mean_squared_error(test, forecast)
    return forecast, mse

def run_prophet(df, test):
    try:
        from prophet import Prophet
    except ImportError:
        st.error("Prophet is not installed. Please add 'prophet' to requirements.txt.")
        return None, None
    df_prophet = df.reset_index()[['Date', 'Close']]
    df_prophet.columns = ['ds', 'y']
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=len(test))
    forecast = model.predict(future)
    yhat = forecast.set_index('ds')['yhat'][-len(test):].values
    mse = mean_squared_error(test, yhat)
    return yhat, mse

# --- Dashboard Tabs ---
tabs = st.tabs(["🏠 Overview", "🤖 Modeling", "📊 Comparison", "💡 Insights"])

# --- Overview Tab ---
with tabs[0]:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:#4F8BF9;'>Overview</h2>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Min Price", f"${df['Close'].min():.2f}")
    col2.metric("Max Price", f"${df['Close'].max():.2f}")
    col3.metric("Mean Price", f"${df['Close'].mean():.2f}")
    col4.metric("30d Volatility", f"{df['Volatility'].iloc[-1]:.2f}")
    st.markdown("<br>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df.index, df['Close'], label='Close', color='#4F8BF9')
    ax.plot(df.index, df['MA_50'], label='50-Day MA', color='#F97B4F', alpha=0.7)
    ax.plot(df.index, df['MA_200'], label='200-Day MA', color='#A14FF9', alpha=0.7)
    ax.set_title('Price Trend with Moving Averages', fontsize=15, color='#4F8BF9')
    ax.legend()
    ax.grid(True, alpha=0.2)
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Modeling Tab ---
with tabs[1]:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:#4F8BF9;'>Modeling</h2>", unsafe_allow_html=True)
    model_choice = st.selectbox(
        "Select Model",
        [model_icons[m] for m in ["ARIMA", "SARIMA", "Prophet"]],
        format_func=lambda x: x,
        key="modeling_selectbox"
    )
    model_choice = model_choice.split(' ', 1)[1]
    if model_choice == "ARIMA":
        forecast, mse = run_arima(train, test)
        st.markdown(f"<h4 style='color:#4F8BF9;'>🔵 ARIMA Test MSE: {mse:.2f}</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(test.index, test.values, label='Actual', color='#222')
        ax.plot(test.index, forecast, label='ARIMA Forecast', color='#4F8BF9')
        ax.set_title('ARIMA Forecast vs Actual', fontsize=16, color='#4F8BF9')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        # Download
        result_df = pd.DataFrame({'Date': test.index, 'Actual': test.values, 'ARIMA Forecast': forecast})
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download ARIMA Results as CSV", csv, "arima_results.csv", "text/csv")
    elif model_choice == "SARIMA":
        forecast, mse = run_sarima(train, test)
        st.markdown(f"<h4 style='color:#A14FF9;'>🟣 SARIMA Test MSE: {mse:.2f}</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(test.index, test.values, label='Actual', color='#222')
        ax.plot(test.index, forecast, label='SARIMA Forecast', color='#A14FF9')
        ax.set_title('SARIMA Forecast vs Actual', fontsize=16, color='#A14FF9')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        result_df = pd.DataFrame({'Date': test.index, 'Actual': test.values, 'SARIMA Forecast': forecast})
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download SARIMA Results as CSV", csv, "sarima_results.csv", "text/csv")
    elif model_choice == "Prophet":
        forecast, mse = run_prophet(df, test)
        if forecast is not None:
            st.markdown(f"<h4 style='color:#F97B4F;'>🟠 Prophet Test MSE: {mse:.2f}</h4>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(test.index, test.values, label='Actual', color='#222')
            ax.plot(test.index, forecast, label='Prophet Forecast', color='#F97B4F')
            ax.set_title('Prophet Forecast vs Actual', fontsize=16, color='#F97B4F')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            result_df = pd.DataFrame({'Date': test.index, 'Actual': test.values, 'Prophet Forecast': forecast})
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Prophet Results as CSV", csv, "prophet_results.csv", "text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

# --- Comparison Tab ---
with tabs[2]:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:#4F8BF9;'>Model Comparison</h2>", unsafe_allow_html=True)
    arima_pred, arima_mse = run_arima(train, test)
    sarima_pred, sarima_mse = run_sarima(train, test)
    prophet_pred, prophet_mse = run_prophet(df, test)
    mse_dict = {"ARIMA": arima_mse, "SARIMA": sarima_mse, "Prophet": prophet_mse}
    best_model = min(mse_dict, key=mse_dict.get)
    summary = pd.DataFrame({
        'Model': ['ARIMA', 'SARIMA', 'Prophet'],
        'MSE': [arima_mse, sarima_mse, prophet_mse]
    })
    st.markdown("<h4 style='color:#4F8BF9;'>✨ Model Comparison Table</h4>", unsafe_allow_html=True)
    st.dataframe(summary, use_container_width=True)
    st.markdown(f"<div class='highlight'><b>Best Model:</b> {best_model} (MSE: {mse_dict[best_model]:.2f})</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test.index, test.values, label='Actual', color='#222')
    ax.plot(test.index, arima_pred, label='ARIMA', color='#4F8BF9')
    ax.plot(test.index, sarima_pred, label='SARIMA', color='#A14FF9')
    if prophet_pred is not None:
        ax.plot(test.index, prophet_pred, label='Prophet', color='#F97B4F')
    ax.set_title('Model Comparison: Last 30 Days', fontsize=16, color='#4F8BF9')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    # Download all results
    comp_df = pd.DataFrame({
        'Date': test.index,
        'Actual': test.values,
        'ARIMA': arima_pred,
        'SARIMA': sarima_pred,
        'Prophet': prophet_pred if prophet_pred is not None else np.nan
    })
    csv = comp_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Comparison Results as CSV", csv, "comparison_results.csv", "text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

# --- Insights Tab ---
with tabs[3]:
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:#4F8BF9;'>Insights</h2>", unsafe_allow_html=True)
    st.markdown(f"<b>Best Model (Lowest MSE):</b> <span style='color:#4F8BF9;font-size:1.2em'>{best_model}</span>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<b>30-Day Rolling Volatility</b>")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df['Volatility'], color='#F97B4F', label='30d Volatility')
    ax.set_title('Rolling Volatility (30 Days)', fontsize=14, color='#F97B4F')
    ax.grid(True, alpha=0.2)
    ax.legend()
    st.pyplot(fig)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<b>Quick Tips:</b>")
    st.markdown("""
    - Use the **Overview** tab to understand the data and trends.
    - Try all models in the **Modeling** tab to see their forecasts.
    - Use the **Comparison** tab to see which model performs best.
    - Download results for further analysis.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<div class='footer'>
    Made with ❤️ using <a href='https://streamlit.io/' target='_blank'>Streamlit</a> | <b>StockSight Dashboard</b> 2024
</div>
""", unsafe_allow_html=True) 