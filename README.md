

An interactive stock market prediction web app built with **Streamlit** and **Machine Learning**. Enter any stock ticker, analyze technical indicators, and get AI-powered predictions on price direction and future trends.


---

## Features

- **Candlestick Chart** with Bollinger Bands, MA 20 and MA 50
- **AI Direction Prediction** — predicts if stock will go UP or DOWN tomorrow
- **Price Forecast** — linear trend forecast for up to 90 days
- **RSI Indicator** — spot overbought and oversold conditions
- **MACD Indicator** — detect momentum and trend changes
- **Feature Importance Chart** — see which indicators influence the model most
- **Raw Data Table** — view the last 50 rows of stock data

---

## 🖥️ Demo

> Enter a ticker like `AAPL`, `TSLA`, `GOOGL`, or Indian stocks like `RELIANCE.NS` → Select history period → Click **🚀 Run Prediction**

---

## 🛠️ Tech Stack

| [Streamlit](https://streamlit.io/) | Web app UI framework |
| [yFinance](https://github.com/ranaroussi/yfinance) | Stock market data fetching |
| [scikit-learn](https://scikit-learn.org/) | Random Forest ML model |
| [Plotly](https://plotly.com/) | Interactive charts |
| [Pandas](https://pandas.pydata.org/) | Data manipulation |
| [NumPy](https://numpy.org/) | Numerical computations |

---

Installation


### 1. Install dependencies
```bash
pip install streamlit yfinance pandas numpy scikit-learn plotly
```

### 2. Run the app(in terminal)
```bash
streamlit run app.py
```

### 3. Open in browser
```
http://localhost:8501
```

---

## 📁 Project Structure

```
ai-stock-predictor/
│
├── app.py              # Main Streamlit application (all-in-one)
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies
```

---

## 📦 Requirements

Create a `requirements.txt` file with the following:

```
streamlit
yfinance
pandas
numpy
scikit-learn
plotly
```

---

## 📖 How to Use

1. **Enter a Stock Ticker** in the sidebar (e.g. `AAPL` for Apple, `TSLA` for Tesla)
2. **Select History Period** — how much past data to train on (1y, 2y, 5y)
3. **Select Forecast Days** — how many days ahead to predict (7–90)
4. Click **🚀 Run Prediction**
5. View charts, prediction, confidence score and indicators

### Ticker Examples

| Company | Ticker |
|---|---|
| Apple | `AAPL` |
| Tesla | `TSLA` |
| Google | `GOOGL` |
| Microsoft | `MSFT` |
| Reliance Industries | `RELIANCE.NS` |
| TCS | `TCS.NS` |
| Infosys | `INFY.NS` |

---

## 🧠 How It Works

### Random Forest Classifier
- Trained on technical indicators: MA 20, MA 50, RSI, MACD, Volume
- Predicts whether the stock will go **UP or DOWN** the next day
- Shows model accuracy and confidence percentage

### Technical Indicators
| Indicator | Use |
|---|---|
| **MA 20 / MA 50** | Identify trend direction |
| **RSI** | Spot overbought (>70) or oversold (<30) conditions |
| **MACD** | Detect momentum shifts and trend reversals |
| **Bollinger Bands** | Measure price volatility |

### Price Forecast
- Uses linear trend analysis on historical closing prices
- Projects price movement for the selected number of future days
- Displayed as a dashed green line extending from historical data

---




