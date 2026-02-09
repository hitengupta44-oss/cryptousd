# ================= IMPORTS =================
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

# ================= SETTINGS =================
SYMBOL = "BTCUSD"
TIMEFRAME = mt5.TIMEFRAME_M1
LOOKBACK = 60
PRED_MINUTES = 10
ROLLING_CANDLES = 600
BACKEND_URL = "https://cryptousdlive-1.onrender.com/update"

RETRAIN_INTERVAL = 30  # minutes

# ================= MT5 =================
if not mt5.initialize():
    raise RuntimeError("MT5 initialization failed")

# ================= MODEL =================
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

# ================= INDICATORS =================
def add_indicators(df):
    df["EMA20"] = df["close"].ewm(span=20).mean()
    df["SMA50"] = df["close"].rolling(50).mean()
    df["RSI"] = ta.momentum.RSIIndicator(df["close"]).rsi()

    macd = ta.trend.MACD(df["close"])
    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()

    df["VWAP"] = (df["close"] * df["tick_volume"]).cumsum() / df["tick_volume"].cumsum()

    return df.dropna()

FEATURES = ["close", "EMA20", "SMA50", "RSI", "MACD", "MACD_SIGNAL", "VWAP"]

# ================= MAIN LOOP =================
while True:
    try:
        # -------- Fetch data --------
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, ROLLING_CANDLES)
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")

        df = add_indicators(df)

        # -------- Prepare ML --------
        data = scaler.fit_transform(df[FEATURES])

        X, y = [], []
        for i in range(LOOKBACK, len(data)):
            X.append(data[i-LOOKBACK:i])
            y.append(data[i, 0])

        X, y = np.array(X), np.array(y)

        # -------- Live retraining --------
        now = datetime.now(timezone.utc)

        if (model is None or last_train_time is None or
            (now - last_train_time).seconds >= RETRAIN_INTERVAL * 60):

            print("Retraining model...")
            model = build_model(len(FEATURES))
            model.fit(X, y, epochs=5, batch_size=32, verbose=0)
            last_train_time = now
            print("Model updated at", now)

        # -------- Prediction --------
        last_seq = X[-1]
        temp_df = df.copy()
        last_time = temp_df.iloc[-1]["time"]

        predictions = []

        for i in range(PRED_MINUTES):
            pred_scaled = model.predict(
                last_seq.reshape(1, LOOKBACK, len(FEATURES)),
                verbose=0
            )[0][0]

            pred_price = scaler.inverse_transform(
                np.hstack([[pred_scaled], np.zeros(len(FEATURES)-1)]).reshape(1, -1)
            )[0][0]

            prev_close = temp_df.iloc[-1]["close"]

            future_time = last_time + timedelta(minutes=i+1)

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

            latest_features = temp_df[FEATURES].tail(LOOKBACK)
            last_seq = scaler.transform(latest_features)

            predictions.append(temp_df.iloc[-1])

        # -------- Send REAL --------
        last = df.iloc[-1]

        real_payload = {
            "time": last["time"].isoformat(),
            "open": float(last["open"]),
            "high": float(last["high"]),
            "low": float(last["low"]),
            "close": float(last["close"]),
            "ema20": float(last["EMA20"]),
            "sma50": float(last["SMA50"]),
            "vwap": float(last["VWAP"]),
            "rsi": float(last["RSI"]),
            "signal": "BUY" if last["EMA20"] > last["SMA50"] else "SELL",
            "type": "real"
        }

        requests.post(BACKEND_URL, json=real_payload)

        # -------- Send predictions --------
        prev_close = last["close"]

        for row in predictions:
            signal = "BUY" if row["close"] > prev_close else "SELL"

            payload = {
                "time": row["time"].isoformat(),
                "open": float(prev_close),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "signal": signal,
                "type": "prediction"
            }

            prev_close = row["close"]
            requests.post(BACKEND_URL, json=payload)

        print("Sent real + predictions at", last_time)

    except Exception as e:
        print("Error:", e)

    time.sleep(60)
