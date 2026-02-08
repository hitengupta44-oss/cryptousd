import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import requests
import time
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import ta

SYMBOL = "BTCUSD"
TIMEFRAME = mt5.TIMEFRAME_M1
LOOKBACK = 60
PRED_MINUTES = 10
BACKEND_URL = "https://cryptousd.onrender.com/update"

if not mt5.initialize():
    raise RuntimeError("MT5 init failed")

def build_model(n_features):
    model = Sequential([
        LSTM(64, input_shape=(LOOKBACK, n_features)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

model = None
scaler = MinMaxScaler()

while True:
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 500)
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    # ===== INDICATORS =====
    df["EMA20"] = df["close"].ewm(span=20).mean()
    df["SMA50"] = df["close"].rolling(50).mean()
    df["VWAP"] = (df["close"] * df["tick_volume"]).cumsum() / df["tick_volume"].cumsum()
    df["RSI"] = ta.momentum.RSIIndicator(df["close"]).rsi()
    macd = ta.trend.MACD(df["close"])
    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()

    df.dropna(inplace=True)

    features = ["close", "EMA20", "SMA50", "VWAP", "RSI", "MACD", "MACD_SIGNAL"]
    data = scaler.fit_transform(df[features])

    X, y = [], []
    for i in range(LOOKBACK, len(data)):
        X.append(data[i-LOOKBACK:i])
        y.append(data[i, 0])

    X, y = np.array(X), np.array(y)

    if model is None:
        model = build_model(len(features))
        model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    last_seq = X[-1]
    last_price = df.iloc[-1]["close"]
    last_time = df.iloc[-1]["time"]

    # ===== MULTI-STEP FORECAST =====
    pred_prices = []
    seq = last_seq.copy()

    for _ in range(PRED_MINUTES):
        p = model.predict(seq.reshape(1, LOOKBACK, len(features)), verbose=0)[0][0]
        price = scaler.inverse_transform(
            np.hstack([[p], np.zeros(len(features)-1)]).reshape(1, -1)
        )[0][0]

        pred_prices.append(price)

        new_row = np.hstack([[p], seq[-1][1:]])
        seq = np.vstack([seq[1:], new_row])

    # ===== REAL CANDLE =====
    last = df.iloc[-1]
    real_candle = {
        "time": last_time.isoformat(),
        "open": float(last["open"]),
        "high": float(last["high"]),
        "low": float(last["low"]),
        "close": float(last["close"]),
        "signal": "BUY" if pred_prices[0] > last_price else "SELL",
        "type": "real"
    }

    # ===== FUTURE CANDLES =====
    future_candles = []
    prev_close = last_price

    for i, price in enumerate(pred_prices):
        t = last_time + timedelta(minutes=i+1)
        future_candles.append({
            "time": t.isoformat(),
            "open": prev_close,
            "high": max(prev_close, price),
            "low": min(prev_close, price),
            "close": price,
            "signal": "BUY" if price > prev_close else "SELL",
            "type": "prediction"
        })
        prev_close = price

    payload = {
        "type": "prediction",
        "candles": [real_candle] + future_candles
    }

    requests.post(BACKEND_URL, json=payload, timeout=5)
    print("Sent real + 10 future candles")

    time.sleep(60)
