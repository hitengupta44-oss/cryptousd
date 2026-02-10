# ================= FINAL PRODUCTION PRODUCER =================

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
last_processed_time = None

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

print("Producer started... Waiting for new candles")

# ================= MAIN LOOP =================
while True:
    try:
        # ---- Get last CLOSED candles (skip forming candle) ----
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 1, ROLLING_CANDLES)
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")

        current_time = df.iloc[-1]["time"]

        # ---- Wait until a NEW candle arrives ----
        if last_processed_time == current_time:
            time.sleep(2)
            continue

        last_processed_time = current_time
        print("New candle:", current_time)

        # ================= DATA PROCESSING =================
        df = add_indicators(df)

        volatility = df["close"].pct_change().std()

        scaled = scaler.fit_transform(df[FEATURES])

        X, y = [], []
        for i in range(LOOKBACK, len(scaled)):
            X.append(scaled[i-LOOKBACK:i])
            y.append(scaled[i, 0])

        X, y = np.array(X), np.array(y)

        # ---- Build model first time ----
        if model is None:
            print("Building model...")
            model = build_model(len(FEATURES))

        # ---- Live retraining ----
        now = datetime.now(timezone.utc)
        if (last_train_time is None or
            (now - last_train_time).total_seconds() >= RETRAIN_INTERVAL * 60):

            print("Updating model...")
            model.fit(X, y, epochs=3, batch_size=32, verbose=0)
            last_train_time = now
            print("Model updated at", now)

        # ================= PREDICTIONS =================
        base_time = current_time
        last_seq = X[-1]
        temp_df = df.copy()
        predictions = []

        for i in range(PRED_MINUTES):
            pred_scaled = model.predict(
                last_seq.reshape(1, LOOKBACK, len(FEATURES)),
                verbose=0
            )[0][0]

            pred_price = scaler.inverse_transform(
                np.hstack([[pred_scaled], np.zeros(len(FEATURES)-1)]).reshape(1, -1)
            )[0][0]

            # Add realistic market noise
            pred_price = pred_price * (1 + np.random.normal(0, volatility))

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

            latest_features = temp_df[FEATURES].tail(LOOKBACK)
            last_seq = scaler.transform(latest_features)

            predictions.append(temp_df.iloc[-1])

        # ================= SEND REAL =================
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

        # ================= SEND PREDICTIONS =================
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

        print("Sent real + predictions at", base_time)

    except Exception as e:
        print("Error:", e)

    time.sleep(2)
