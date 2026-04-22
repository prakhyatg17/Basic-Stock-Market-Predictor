import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # Moving Averages
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))

    # MACD
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26

    # Bollinger Bands
    df["BB_upper"] = df["MA_20"] + 2 * df["Close"].rolling(20).std()
    df["BB_lower"] = df["MA_20"] - 2 * df["Close"].rolling(20).std()

    # Target: next-day direction
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    return df.dropna()