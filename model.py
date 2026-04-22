from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from prophet import Prophet
import pandas as pd

FEATURES = ["MA_20", "MA_50", "RSI", "MACD", "Volume"]

def train_classifier(df):
    X = df[FEATURES]
    y = df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

def forecast_price(df, days=30):
    prophet_df = df[["Close"]].reset_index()
    prophet_df.columns = ["ds", "y"]
    prophet_df["ds"] = prophet_df["ds"].dt.tz_localize(None)
    m = Prophet(daily_seasonality=True)
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=days)
    forecast = m.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]