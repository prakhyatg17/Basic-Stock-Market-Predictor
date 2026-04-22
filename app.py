import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="AI Stock Predictor", layout="wide")
st.title("📈 AI Stock Market Predictor")
st.info("👈 Enter a stock ticker in the sidebar and click **Run Prediction** to start.")

# ---- SIDEBAR ----
ticker = st.sidebar.text_input("Stock Ticker (e.g. AAPL, TSLA, GOOGL)", value="AAPL").upper()
period = st.sidebar.selectbox("History Period", ["1y", "2y", "5y"], index=1)
forecast_days = st.sidebar.slider("Forecast Days", 7, 90, 30)
run = st.sidebar.button("🚀 Run Prediction")

# ---- HELPER FUNCTIONS ----

def fetch_data(ticker, period):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df.dropna(inplace=True)
    return df

def add_features(df):
    df = df.copy()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))

    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26

    df["BB_upper"] = df["MA_20"] + 2 * df["Close"].rolling(20).std()
    df["BB_lower"] = df["MA_20"] - 2 * df["Close"].rolling(20).std()

    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df.dropna()

def train_model(df):
    features = ["MA_20", "MA_50", "RSI", "MACD", "Volume"]
    X = df[features]
    y = df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc, features

def simple_forecast(df, days):
    # Linear trend forecast (no Prophet needed)
    closes = df["Close"].values
    x = np.arange(len(closes))
    coeffs = np.polyfit(x, closes, 1)
    future_x = np.arange(len(closes), len(closes) + days)
    forecast = np.polyval(coeffs, future_x)

    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date, periods=days + 1, freq="B")[1:]
    return future_dates, forecast

# ---- MAIN APP ----

if run:
    with st.spinner(f"Fetching data for {ticker}..."):
        try:
            df_raw = fetch_data(ticker, period)
            if df_raw.empty:
                st.error(f"Could not find data for ticker '{ticker}'. Please check the symbol and try again.")
                st.stop()
            df = add_features(df_raw)
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.stop()

    # ---- PRICE CHART ----
    st.subheader(f"📊 {ticker} Price History")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA_20"], name="MA 20", line=dict(color="orange", width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA_50"], name="MA 50", line=dict(color="blue", width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Upper", line=dict(color="gray", dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Lower", line=dict(color="gray", dash="dot")))
    fig.update_layout(xaxis_rangeslider_visible=False, height=500)
    st.plotly_chart(fig, use_container_width=True)

    # ---- PREDICTION ----
    st.subheader("🤖 Tomorrow's Direction Prediction")
    with st.spinner("Training model..."):
        model, acc, features = train_model(df)
        latest = df[features].iloc[-1:]
        prediction = model.predict(latest)[0]
        prob = model.predict_proba(latest)[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Model Accuracy", f"{acc:.1%}")
    col2.metric("Prediction", "📈 UP" if prediction == 1 else "📉 DOWN")
    col3.metric("Confidence", f"{max(prob):.1%}")

    # Feature importance
    st.subheader("🔍 Feature Importance")
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)
    fig_imp = go.Figure(go.Bar(
        x=importance_df["Importance"],
        y=importance_df["Feature"],
        orientation="h",
        marker_color="steelblue"
    ))
    fig_imp.update_layout(height=300)
    st.plotly_chart(fig_imp, use_container_width=True)

    # ---- FORECAST ----
    st.subheader(f"🔮 {forecast_days}-Day Price Forecast (Trend)")
    future_dates, forecast = simple_forecast(df, forecast_days)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Historical", line=dict(color="steelblue")))
    fig2.add_trace(go.Scatter(x=future_dates, y=forecast, name="Forecast", line=dict(color="green", dash="dash")))
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)

    # ---- RSI ----
    st.subheader("📉 RSI Indicator")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="purple")))
    fig3.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
    fig3.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    fig3.update_layout(height=300)
    st.plotly_chart(fig3, use_container_width=True)

    # ---- MACD ----
    st.subheader("📊 MACD Indicator")
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="orange")))
    fig4.add_hline(y=0, line_dash="dash", line_color="gray")
    fig4.update_layout(height=300)
    st.plotly_chart(fig4, use_container_width=True)

    # ---- RAW DATA ----
    with st.expander("📄 View Raw Data"):
        st.dataframe(df[["Open", "High", "Low", "Close", "Volume", "MA_20", "MA_50", "RSI", "MACD"]].tail(50))

    st.warning("⚠️ This tool is for educational purposes only. Never make financial decisions based solely on AI predictions.")
