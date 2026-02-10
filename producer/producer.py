import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta, timezone
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import ta

# SETTINGS
SYMBOL = "BTCUSD"
TIMEFRAME = mt5.TIMEFRAME_M1
LOOKBACK = 60
PRED_MINUTES = 10
ROLLING_CANDLES = 1000
BACKEND_URL = "https://cryptousdlive-1.onrender.com/update"
RETRAIN_INTERVAL = 30

if not mt5.initialize():
    raise RuntimeError("MT5 initialization failed")

def build_model(n_features):
    model = Sequential([
        Input(shape=(LOOKBACK, n_features)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

model = None
scaler = MinMaxScaler()
last_train_time = None
last_processed_time = None

FEATURES = ["close","EMA20","SMA50","RSI","MACD","MACD_SIGNAL","VWAP"]

def add_indicators(df):
    df["EMA20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["SMA50"] = df["close"].rolling(50).mean()
    df["RSI"] = ta.momentum.RSIIndicator(df["close"]).rsi()

    macd = ta.trend.MACD(df["close"])
    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()

    df["VWAP"] = (df["close"] * df["tick_volume"]).cumsum() / df["tick_volume"].cumsum()

    return df.dropna()

def create_sequences(data):
    X, y = [], []
    for i in range(LOOKBACK, len(data)):
        X.append(data[i-LOOKBACK:i])
        y.append(data[i,0])
    return np.array(X), np.array(y)

print("Producer started")

while True:
    try:
        # Closed candles only
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 1, ROLLING_CANDLES)
        df = pd.DataFrame(rates)

        if df.empty:
            time.sleep(5)
            continue

        df["time"] = pd.to_datetime(df["time"], unit="s")
        current_time = df.iloc[-1]["time"]

        # Wait for new candle
        if last_processed_time == current_time:
            time.sleep(2)
            continue

        last_processed_time = current_time
        print("New candle:", current_time)

        df = add_indicators(df)

        if len(df) < LOOKBACK + 20:
            continue

        scaled = scaler.fit_transform(df[FEATURES])
        X, y = create_sequences(scaled)

        if model is None:
            model = build_model(len(FEATURES))

        now = datetime.now(timezone.utc)

        if last_train_time is None or (now - last_train_time).total_seconds() > RETRAIN_INTERVAL*60:
            model.fit(X, y, epochs=5, batch_size=32, verbose=0)
            last_train_time = now
            print("Model trained")

        # ===== Predictions =====
        base_time = current_time
        last_seq = X[-1]
        temp_df = df.copy()
        predictions = []

        volatility = df["close"].pct_change().std()

        for i in range(PRED_MINUTES):

            pred_scaled = model.predict(last_seq.reshape(1,LOOKBACK,len(FEATURES)), verbose=0)[0][0]

            pred_price = scaler.inverse_transform(
                np.hstack([[pred_scaled], np.zeros(len(FEATURES)-1)]).reshape(1,-1)
            )[0][0]

            pred_price *= (1 + np.random.normal(0, volatility))

            prev_close = temp_df.iloc[-1]["close"]

            future_time = base_time + timedelta(minutes=i+1)

            new_row = {
                "time": future_time,
                "open": prev_close,
                "high": max(prev_close, pred_price),
                "low": min(prev_close, pred_price),
                "close": pred_price,
                "tick_volume": temp_df.iloc[-1]["tick_volume"]
            }

            temp_df = pd.concat([temp_df, pd.DataFrame([new_row])], ignore_index=True)
            temp_df = add_indicators(temp_df)

            last_seq = scaler.transform(temp_df[FEATURES].tail(LOOKBACK))
            predictions.append(new_row)

        # ===== Send REAL =====
        last = df.iloc[-1]

        signal = None
        if df.iloc[-1]["EMA20"] > df.iloc[-2]["SMA50"] and df.iloc[-2]["EMA20"] <= df.iloc[-2]["SMA50"]:
            signal = "BUY"
        elif df.iloc[-1]["EMA20"] < df.iloc[-2]["SMA50"] and df.iloc[-2]["EMA20"] >= df.iloc[-2]["SMA50"]:
            signal = "SELL"

        requests.post(BACKEND_URL, json={
            "time": last["time"].isoformat(),
            "open": float(last["open"]),
            "high": float(last["high"]),
            "low": float(last["low"]),
            "close": float(last["close"]),
            "ema20": float(last["EMA20"]),
            "sma50": float(last["SMA50"]),
            "vwap": float(last["VWAP"]),
            "rsi": float(last["RSI"]),
            "signal": signal,
            "type": "real"
        })

        # ===== Send Predictions =====
        for row in predictions:
            requests.post(BACKEND_URL, json={
                "time": pd.Timestamp(row["time"]).isoformat(),
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "type": "prediction"
            })

        print("Sent 60 real + 10 future")

    except Exception as e:
        print("Error:", e)

    time.sleep(2)
