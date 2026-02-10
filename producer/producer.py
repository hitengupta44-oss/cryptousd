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

BACKEND_URL = "https://cryptousdlive-1.onrender.com/update"

SYMBOL = "BTCUSD"
TIMEFRAME = mt5.TIMEFRAME_M1
LOOKBACK = 60
PRED_MINUTES = 10


if not mt5.initialize():
    print("MT5 failed")
    quit()

model = None
last_candle_time = None

while True:
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)

        rates = mt5.copy_rates_range(SYMBOL, TIMEFRAME, start_time, end_time)
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")

        # Only closed candle
        current_time = df.iloc[-1]["time"]

        if last_candle_time == current_time:
            time.sleep(5)
            continue

        last_candle_time = current_time
        print("New candle:", current_time)

        # Indicators
        df["EMA20"] = df["close"].ewm(span=20).mean()
        df["SMA50"] = df["close"].rolling(50).mean()
        df["RSI"] = ta.momentum.RSIIndicator(df["close"]).rsi()

        macd = ta.trend.MACD(df["close"])
        df["MACD"] = macd.macd()
        df["MACD_SIGNAL"] = macd.macd_signal()

        df["VWAP"] = (df["close"] * df["tick_volume"]).cumsum() / df["tick_volume"].cumsum()

        df = df.dropna()

        features = ["close","EMA20","SMA50","RSI","MACD","MACD_SIGNAL","VWAP"]
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[features])

        # Create sequences
        X, y = [], []
        for i in range(LOOKBACK, len(scaled)):
            X.append(scaled[i-LOOKBACK:i])
            y.append(scaled[i,0])

        X, y = np.array(X), np.array(y)

        # Build model if first run
        if model is None:
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(LOOKBACK, len(features))),
                Dropout(0.2),
                LSTM(32),
                Dense(16, activation="relu"),
                Dense(1)
            ])
            model.compile(optimizer="adam", loss="mse")

        # Train every minute on last 24h
        model.fit(X, y, epochs=2, batch_size=32, verbose=0)

        # ===== Send last 60 real candles =====
        real_60 = df.tail(60)

        for _, row in real_60.iterrows():
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

        # ===== Predictions =====
        last_seq = scaled[-LOOKBACK:]
        temp_df = df.copy()

        for i in range(PRED_MINUTES):
            pred_scaled = model.predict(last_seq.reshape(1,LOOKBACK,len(features)), verbose=0)[0][0]

            pred_price = scaler.inverse_transform(
                np.hstack([[pred_scaled], np.zeros(len(features)-1)]).reshape(1,-1)
            )[0][0]

            prev_close = temp_df.iloc[-1]["close"]
            future_time = temp_df.iloc[-1]["time"] + timedelta(minutes=1)

            new_row = {
                "time": future_time,
                "open": prev_close,
                "high": max(prev_close, pred_price),
                "low": min(prev_close, pred_price),
                "close": pred_price,
                "tick_volume": temp_df.iloc[-1]["tick_volume"]
            }

            temp_df = pd.concat([temp_df, pd.DataFrame([new_row])])

            # Recalculate indicators for predictions
            temp_df["EMA20"] = temp_df["close"].ewm(span=20).mean()
            temp_df["SMA50"] = temp_df["close"].rolling(50).mean()
            temp_df["RSI"] = ta.momentum.RSIIndicator(temp_df["close"]).rsi()
            temp_df["VWAP"] = (temp_df["close"] * temp_df["tick_volume"]).cumsum() / temp_df["tick_volume"].cumsum()

            latest = temp_df.iloc[-1]

            payload = {
                "time": latest["time"].isoformat(),
                "open": float(latest["open"]),
                "high": float(latest["high"]),
                "low": float(latest["low"]),
                "close": float(latest["close"]),
                "ema20": float(latest["EMA20"]),
                "sma50": float(latest["SMA50"]),
                "vwap": float(latest["VWAP"]),
                "rsi": float(latest["RSI"]),
                "type": "prediction"
            }

            requests.post(BACKEND_URL, json=payload)

            last_seq = scaler.transform(temp_df[features].tail(LOOKBACK))

    except Exception as e:
        print("Error:", e)

    time.sleep(10)
