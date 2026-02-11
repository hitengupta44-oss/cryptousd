import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import ta
from binance.client import Client

# ================= CONFIG =================

BACKEND_URL = "https://cryptoliveusdt.onrender.com/update"
SYMBOL = "BTCUSDT"
INTERVAL = Client.KLINE_INTERVAL_1MINUTE

LOOKBACK = 60
PRED_MINUTES = 10

client = Client()
model = None
scaler = MinMaxScaler()
last_candle_time = None

print("Institutional Producer Started")

# ==========================================

def fetch_24h_data():
    klines = client.get_klines(symbol=SYMBOL, interval=INTERVAL, limit=1440)

    df = pd.DataFrame(klines, columns=[
        'open_time','open','high','low','close','volume',
        'close_time','qav','trades','taker_base','taker_quote','ignore'
    ])

    df['time'] = pd.to_datetime(df['open_time'], unit='ms')
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)

    return df[['time','open','high','low','close','volume']]


def build_model(n_features):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LOOKBACK, n_features)),
        Dropout(0.2),
        LSTM(32),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


# ==========================================

while True:
    try:
        df = fetch_24h_data()

        current_time = df.iloc[-1]["time"]

        # Wait for new closed candle
        if last_candle_time == current_time:
            time.sleep(5)
            continue

        last_candle_time = current_time
        print("New candle:", current_time)

        # ================= INDICATORS =================
        df["EMA20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["SMA50"] = df["close"].rolling(50).mean()
        df["RSI"] = ta.momentum.RSIIndicator(df["close"]).rsi()

        macd = ta.trend.MACD(df["close"])
        df["MACD"] = macd.macd()
        df["MACD_SIGNAL"] = macd.macd_signal()

        df["VWAP"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

        # ===== Institutional stability =====
        df["RET"] = df["close"].pct_change()

        df = df.dropna()

        features = ["RET","EMA20","SMA50","RSI","MACD","MACD_SIGNAL","VWAP"]
        scaled = scaler.fit_transform(df[features])

        # ================= SEQUENCES =================
        X, y = [], []
        for i in range(LOOKBACK, len(scaled)):
            X.append(scaled[i-LOOKBACK:i])
            y.append(scaled[i, 0])

        X, y = np.array(X), np.array(y)

        # ================= MODEL =================
        if model is None:
            model = build_model(len(features))
            model.fit(X, y, epochs=8, batch_size=32, verbose=0)
            print("Initial training")
        else:
            model.fit(X[-300:], y[-300:], epochs=1, batch_size=32, verbose=0)

        # ================= SEND REAL 60 =================
        real60 = df.tail(60)

        for _, row in real60.iterrows():
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
                "signal": None,
                "type": "real"
            }
            requests.post(BACKEND_URL, json=payload)

        # ================= PREDICTIONS =================
        last_seq = scaled[-LOOKBACK:]
        last_price = df.iloc[-1]["close"]
        last_return = df.iloc[-1]["RET"]
        volatility = df["RET"].std()

        temp_df = df.copy()

        for i in range(PRED_MINUTES):

            pred_scaled = model.predict(
                last_seq.reshape(1, LOOKBACK, len(features)), verbose=0
            )[0][0]

            # Smooth prediction
            pred_scaled = 0.6 * pred_scaled + 0.4 * last_return

            pred_return = scaler.inverse_transform(
                np.hstack([[pred_scaled], np.zeros(len(features)-1)]).reshape(1,-1)
            )[0][0]

            # Limit extreme moves
            pred_return = np.clip(pred_return, -1.5*volatility, 1.5*volatility)

            pred_price = last_price * (1 + pred_return)

            # Realistic candle (with wicks)
            open_p = last_price
            close_p = pred_price
            wick = abs(pred_return) * last_price * 0.5

            high_p = max(open_p, close_p) + wick
            low_p = min(open_p, close_p) - wick

            future_time = current_time + timedelta(minutes=i+1)

            new_row = {
                "time": future_time,
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
                "volume": temp_df.iloc[-1]["volume"]
            }

            temp_df = pd.concat([temp_df, pd.DataFrame([new_row])], ignore_index=True)

            # Recalculate indicators
            temp_df["EMA20"] = temp_df["close"].ewm(span=20, adjust=False).mean()
            temp_df["SMA50"] = temp_df["close"].rolling(50).mean()
            temp_df["RSI"] = ta.momentum.RSIIndicator(temp_df["close"]).rsi()
            temp_df["VWAP"] = (temp_df["close"] * temp_df["volume"]).cumsum() / temp_df["volume"].cumsum()

            latest = temp_df.iloc[-1]
            prev = temp_df.iloc[-2]

            # Crossover signal only
            signal = None
            if latest["EMA20"] > latest["SMA50"] and prev["EMA20"] <= prev["SMA50"]:
                signal = "BUY"
            elif latest["EMA20"] < latest["SMA50"] and prev["EMA20"] >= prev["SMA50"]:
                signal = "SELL"

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
                "signal": signal,
                "type": "prediction"
            }

            requests.post(BACKEND_URL, json=payload)

            last_price = close_p
            last_return = pred_return
            last_seq = scaler.transform(temp_df[features].tail(LOOKBACK))

        print("Sent 60 real + 10 stable predictions")

    except Exception as e:
        print("Error:", e)

    time.sleep(5)
