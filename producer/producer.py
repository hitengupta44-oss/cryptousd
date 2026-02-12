import pandas as pd
import numpy as np
import requests
import time
import os
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import ta
from binance.client import Client

# ===== Flask keep-alive (Render) =====
from flask import Flask
import threading

app = Flask(__name__)

@app.route("/")
def home():
    return "Producer running"

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

threading.Thread(target=run_flask).start()

# ================= CONFIG =================

BACKEND_URL = "https://cryptoliveusdt.onrender.com/update"
SYMBOL = "BTCUSDT"
INTERVAL = Client.KLINE_INTERVAL_1MINUTE
LOOKBACK = 60
PRED_MINUTES = 10
RETRAIN_INTERVAL = 1800  # 30 min

client = Client()

print("Producer started — Stable MT5 Model")

model = None
scaler = MinMaxScaler()
last_candle_time = None
last_train_time = None

# ================= MODEL =================

def build_model(n_features):
    model = Sequential([
        Input(shape=(LOOKBACK, n_features)),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# ================= FETCH DATA =================

def fetch_data():
    klines = client.get_klines(
        symbol=SYMBOL,
        interval=INTERVAL,
        limit=500
    )

    df = pd.DataFrame(klines, columns=[
        'open_time','open','high','low','close','volume',
        'close_time','qav','trades','taker_base','taker_quote','ignore'
    ])

    df['time'] = pd.to_datetime(df['open_time'], unit='ms')
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)

    return df[['time','open','high','low','close','volume']]

# ================= MAIN LOOP =================

while True:
    try:
        df = fetch_data()

        # ===== Safety check (Binance failure protection) =====
        if df is None or len(df) < 100:
            print("Data fetch issue, retrying...")
            time.sleep(5)
            continue

        # Use LAST CLOSED candle
        current_time = df.iloc[-2]["time"]

        if last_candle_time == current_time:
            print("Waiting for new candle...")
            time.sleep(5)
            continue

        last_candle_time = current_time
        print("New closed candle:", current_time)

        # ===== Indicators =====
        df["EMA20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["SMA50"] = df["close"].rolling(50).mean()
        df["RSI"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
        df["VWAP"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
        df["RET"] = df["close"].pct_change()

        df = df.dropna()

        features = ["RET","EMA20","SMA50","RSI","VWAP"]
        scaled = scaler.fit_transform(df[features])

        # ===== Sequences =====
        X, y = [], []
        saled = scaled
        for i in range(LOOKBACK, len(saled)):
            X.append(saled[i-LOOKBACK:i])
            y.append(saled[i,0])

        X, y = np.array(X), np.array(y)

        # Safety: avoid training on empty data
        if len(X) == 0:
            print("Not enough data for training")
            time.sleep(5)
            continue

        # ===== Train =====
        if model is None:
            model = build_model(len(features))
            model.fit(X, y, epochs=3, batch_size=32, verbose=0)
            last_train_time = time.time()

        if time.time() - last_train_time > RETRAIN_INTERVAL:
            model.fit(X, y, epochs=1, batch_size=32, verbose=0)
            last_train_time = time.time()

        # ===== Send last 60 real =====
        real60 = df.tail(60).reset_index(drop=True)

        for i, row in real60.iterrows():
            signal = None
            if i > 0:
                prev = real60.iloc[i-1]
                if row["EMA20"] > row["SMA50"] and prev["EMA20"] <= prev["SMA50"]:
                    signal = "BUY"
                elif row["EMA20"] < row["SMA50"] and prev["EMA20"] >= prev["SMA50"]:
                    signal = "SELL"

            requests.post(BACKEND_URL, json={
                "time": row["time"].isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "ema20": float(row["EMA20"]),
                "sma50": float(row["SMA50"]),
                "vwap": float(row["VWAP"]),
                "rsi": float(row["RSI"]),
                "signal": signal,
                "type": "real"
            }, timeout=5)

        # ===== Predictions =====
        volatility = df["RET"].std()
        last_price = df.iloc[-1]["close"]

        last_seq = scaled[-LOOKBACK:]
        temp_df = df.copy()

        for _ in range(PRED_MINUTES):

            pred_scaled = model.predict(
                last_seq.reshape(1, LOOKBACK, len(features)),
                verbose=0
            )[0][0]

            pred_ret = scaler.inverse_transform(
                np.hstack([[pred_scaled], np.zeros(len(features)-1)]).reshape(1,-1)
            )[0][0]

            pred_ret = np.clip(pred_ret, -2.5*volatility, 2.5*volatility)
            pred_price = last_price * (1 + pred_ret)

            # Mean reversion
            ema = temp_df.iloc[-1]["EMA20"]
            pred_price = 0.9 * pred_price + 0.1 * ema

            # Noise
            pred_price *= (1 + np.random.normal(0, volatility/2))

            # Wicks
            body = abs(pred_price - last_price)
            wick = max(body*0.5, last_price*volatility*0.2)

            high_p = max(last_price, pred_price) + wick
            low_p = min(last_price, pred_price) - wick

            future_time = temp_df.iloc[-1]["time"] + timedelta(minutes=1)

            new_row = {
                "time": future_time,
                "open": last_price,
                "high": high_p,
                "low": low_p,
                "close": pred_price,
                "volume": temp_df.iloc[-1]["volume"]
            }

            temp_df = pd.concat([temp_df, pd.DataFrame([new_row])], ignore_index=True)

            temp_df["EMA20"] = temp_df["close"].ewm(span=20).mean()
            temp_df["SMA50"] = temp_df["close"].rolling(50).mean()
            temp_df["RSI"] = ta.momentum.RSIIndicator(temp_df["close"]).rsi()
            temp_df["VWAP"] = (temp_df["close"] * temp_df["volume"]).cumsum() / temp_df["volume"].cumsum()
            temp_df["RET"] = temp_df["close"].pct_change()

            latest = temp_df.iloc[-1]
            prev = temp_df.iloc[-2]

            signal = None
            if latest["EMA20"] > latest["SMA50"] and prev["EMA20"] <= prev["SMA50"]:
                signal = "BUY"
            elif latest["EMA20"] < latest["SMA50"] and prev["EMA20"] >= prev["SMA50"]:
                signal = "SELL"

            requests.post(BACKEND_URL, json={
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
            }, timeout=5)

            last_price = pred_price
            last_seq = scaler.transform(temp_df[features].tail(LOOKBACK))

        print("Sent 60 real + 10 prediction")

    except Exception as e:
        print("Error:", e)

    time.sleep(5)
