import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df.dropna(inplace=True)
    return df