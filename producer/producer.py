# ================= FINAL STABLE PRODUCER =================

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import ta

# ================= SETTINGS =================
BACKEND_URL = "https://cryptousdlive-1.onrender.com/update"

SYMBOL = "BTCUSD"
TIMEFRAME = mt5.TIMEFRAME_M1
LOOKBACK = 60
PRED_MINUTES = 10
TRAIN_HOURS = 24

# ================= MT5 =================
if not mt5.initialize():
    print("MT5 initialization failed")
    quit()

model = None
last_candle_time = None

print("Producer started...")

# ================= MAIN LOOP =================
while True:
    try:
        # -------- Fetch last 24 hours --------
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=TRAIN_HOURS)

        rates = mt5.copy_rates_range(SYMBOL, TIMEFRAME, start_time, end_time)
        if rates is None or len(rates) < 100:
            time.sleep(10)
            continue

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")

        # -------- Only process new CLOSED candle --------
        current_time = df.iloc[-1]["time"]

        if last_candle_time == current_time:
            time.sleep(5)
            continue

        last_candle_time = current_time
        print("New candle:", current_time)

        # ================= INDICATORS =================
        df["EMA20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["SMA50"] = df["close"].rolling(50).mean()
        df["RSI"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

        macd = ta.trend.MACD(df["close"])
        df["MACD"] = macd.macd()
        df["MACD_SIGNAL"] = macd.macd_signal()

        df["VWAP"] = (
            df["close"] * df["tick_volume"]
        ).cumsum() / df["tick_volume"].cumsum()

        df = df.dropna().reset_index(drop=True)

        # ================= ML DATA =================
        features = [
            "close",
            "EMA20",
            "SMA50",
            "RSI",
            "MACD",
            "MACD_SIGNAL",
            "VWAP"
        ]

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[features].values)

        # Create sequences
        X, y = [], []
        for i in range(LOOKBACK, len(scaled)):
            X.append(scaled[i-LOOKBACK:i])
            y.append(scaled[i, 0])

        X, y = np.array(X), np.array(y)

        # ================= MODEL =================
        if model is None:
            model = Sequential([
                LSTM(64, return_sequences=True,
                     input_shape=(LOOKBACK, len(features))),
                Dropout(0.2),
                LSTM(32),
                Dense(16, activation="relu"),
                Dense(1)
            ])
            model.compile(optimizer="adam", loss="mse")
            print("Initial training...")

        # Retrain every new candle on last 24h
        model.fit(X, y, epochs=5, batch_size=32, verbose=0)

        # ================= SEND LAST 60 REAL =================
        for _, row in df.tail(60).iterrows():
            payload = {
                "time": row["time"].isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "ema20": float(row["EMA20"]),
                "sma50": float(row["SMA50"]),
                "vwap": float(row["VWAP"]),
                "rsi": float(row["RSI"]),
                "type": "real"
            }
            requests.post(BACKEND_URL, json=payload)

        # ================= FUTURE PREDICTIONS =================
        last_seq = scaled[-LOOKBACK:].copy()
        temp_df = df.copy()
        pred_rows = []

        for step in range(PRED_MINUTES):

            # Predict next close (scaled)
            pred_scaled = model.predict(
                last_seq.reshape(1, LOOKBACK, len(features)),
                verbose=0
            )[0][0]

            # Convert to real price
            pred_close = scaler.inverse_transform(
                np.hstack([[pred_scaled], np.zeros(len(features)-1)]).reshape(1, -1)
            )[0][0]

            prev_close = temp_df["close"].iloc[-1]
            future_time = temp_df["time"].iloc[-1] + timedelta(minutes=1)

            # Realistic candle
            high = max(prev_close, pred_close) * (1 + np.random.uniform(0.0003, 0.001))
            low = min(prev_close, pred_close) * (1 - np.random.uniform(0.0003, 0.001))

            new_row = {
                "time": future_time,
                "open": prev_close,
                "high": high,
                "low": low,
                "close": pred_close,
                "tick_volume": temp_df["tick_volume"].iloc[-1]
            }

            temp_df = pd.concat([temp_df, pd.DataFrame([new_row])], ignore_index=True)

            # Recalculate indicators (IMPORTANT)
            temp_df["EMA20"] = temp_df["close"].ewm(span=20, adjust=False).mean()
            temp_df["SMA50"] = temp_df["close"].rolling(50).mean()
            temp_df["RSI"] = ta.momentum.RSIIndicator(temp_df["close"], window=14).rsi()

            macd = ta.trend.MACD(temp_df["close"])
            temp_df["MACD"] = macd.macd()
            temp_df["MACD_SIGNAL"] = macd.macd_signal()

            temp_df["VWAP"] = (
                temp_df["close"] * temp_df["tick_volume"]
            ).cumsum() / temp_df["tick_volume"].cumsum()

            latest = temp_df.iloc[-1]
            pred_rows.append(latest)

            # Update sequence
            latest_features = temp_df[features].tail(LOOKBACK).values
            last_seq = scaler.transform(latest_features)

        # ================= SEND FUTURE =================
        for row in pred_rows:
            payload = {
                "time": row["time"].isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "ema20": float(row["EMA20"]),
                "sma50": float(row["SMA50"]),
                "vwap": float(row["VWAP"]),
                "rsi": float(row["RSI"]),
                "type": "prediction"
            }
            requests.post(BACKEND_URL, json=payload)

        print("Sent 60 real + 10 future")

    except Exception as e:
        print("Error:", e)

    time.sleep(5)
