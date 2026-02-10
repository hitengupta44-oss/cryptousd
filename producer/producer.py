import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import ta

# ===== SETTINGS =====
SYMBOL = "BTCUSD"
TIMEFRAME = mt5.TIMEFRAME_M1
LOOKBACK = 60
PRED_MINUTES = 10
ROLLING_CANDLES = 600
BACKEND_URL = "https://cryptousdlive-1.onrender.com/update"

# ===== MT5 =====
if not mt5.initialize():
    raise RuntimeError("MT5 failed")

last_processed_time = None
initial_loaded = False

# ===== MODEL =====
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

# ===== INDICATORS =====
def add_indicators(df):
    df["EMA20"] = df["close"].ewm(span=20).mean()
    df["SMA50"] = df["close"].rolling(50).mean()
    df["RSI"] = ta.momentum.RSIIndicator(df["close"]).rsi()

    macd = ta.trend.MACD(df["close"])
    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()

    df["VWAP"] = (df["close"] * df["tick_volume"]).cumsum() / df["tick_volume"].cumsum()

    return df.dropna()

FEATURES = ["close","EMA20","SMA50","RSI","MACD","MACD_SIGNAL","VWAP"]

print("Producer started")

while True:
    try:
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 1, ROLLING_CANDLES)
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")

        df = add_indicators(df)

        current_time = df.iloc[-1]["time"]

        # ===== INITIAL LOAD (last 60 candles) =====
        if not initial_loaded:
            history = df.tail(60)
            for _, row in history.iterrows():
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
                    "signal": "BUY" if row["EMA20"] > row["SMA50"] else "SELL",
                    "type": "real"
                }
                requests.post(BACKEND_URL, json=payload)

            initial_loaded = True
            last_processed_time = current_time
            print("Loaded initial 60 candles")
            continue

        # Wait for new candle
        if last_processed_time == current_time:
            time.sleep(2)
            continue

        last_processed_time = current_time
        print("New candle:", current_time)

        # ===== ML =====
        scaled = scaler.fit_transform(df[FEATURES])

        X = []
        for i in range(LOOKBACK, len(scaled)):
            X.append(scaled[i-LOOKBACK:i])
        X = np.array(X)

        if model is None:
            model = build_model(len(FEATURES))
            model.fit(X[:-1], scaled[LOOKBACK:,0], epochs=3, verbose=0)

        base_time = current_time
        last_seq = X[-1]
        temp_close = df.iloc[-1]["close"]

        # ===== Send real candle =====
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

        # ===== Predictions =====
        prev_close = last["close"]

        for i in range(PRED_MINUTES):
            pred_scaled = model.predict(last_seq.reshape(1,LOOKBACK,len(FEATURES)),verbose=0)[0][0]

            pred_price = scaler.inverse_transform(
                np.hstack([[pred_scaled], np.zeros(len(FEATURES)-1)]).reshape(1,-1)
            )[0][0]

            future_time = base_time + timedelta(minutes=i+1)

            payload = {
                "time": future_time.isoformat(),
                "open": float(prev_close),
                "high": float(max(prev_close, pred_price)),
                "low": float(min(prev_close, pred_price)),
                "close": float(pred_price),
                "signal": "BUY" if pred_price > prev_close else "SELL",
                "type": "prediction"
            }

            prev_close = pred_price
            requests.post(BACKEND_URL, json=payload)

    except Exception as e:
        print("Error:", e)

    time.sleep(2)
