import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import ta
from binance.client import Client

# ===== Flask (Render keep-alive) =====
from flask import Flask
import threading

app = Flask(__name__)

@app.route("/")
def home():
    return "Producer running"

def run_flask():
    app.run(host="0.0.0.0", port=10000)

threading.Thread(target=run_flask).start()

# ================= CONFIG =================
BACKEND_URL = "https://cryptoliveusdt.onrender.com/update"

SYMBOL = "BTCUSDT"
INTERVAL = Client.KLINE_INTERVAL_1MINUTE
LOOKBACK = 60
PRED_MINUTES = 10
TRAIN_HOURS = 24
RETRAIN_INTERVAL = 30 * 60

# ================= BINANCE =================
client = Client()

print("Producer started — MT5 Logic Stable")

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
def fetch_24h():
    klines = client.get_klines(
        symbol=SYMBOL,
        interval=INTERVAL,
        limit=TRAIN_HOURS * 60
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
        df = fetch_24h()
        current_time = df.iloc[-1]["time"]

        # ===== Run ONLY on new closed candle =====
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

        df["VWAP"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
        df["RET"] = df["close"].pct_change()

        df = df.dropna()

        features = ["RET","EMA20","SMA50","RSI","MACD","MACD_SIGNAL","VWAP"]

        # ================= SCALE =================
        scaled = scaler.fit_transform(df[features])

        # ================= SEQUENCES =================
        X, y = [], []
        for i in range(LOOKBACK, len(saled := scaled)):
            X.append(saled[i-LOOKBACK:i])
            y.append(saled[i, 0])

        X, y = np.array(X), np.array(y)

        if len(X) < 10:
            time.sleep(10)
            continue

        # ================= MODEL =================
        if model is None:
            model = build_model(len(features))
            print("Initial training...")
            model.fit(X, y, epochs=10, batch_size=32, verbose=0)
            last_train_time = time.time()

        if time.time() - last_train_time > RETRAIN_INTERVAL:
            print("Retraining model...")
            model.fit(X, y, epochs=3, batch_size=32, verbose=0)
            last_train_time = time.time()

        # ================= SEND LAST 60 REAL =================
        real_60 = df.tail(60).reset_index(drop=True)

        for idx, row in real_60.iterrows():
            signal = None
            if idx > 0:
                prev = real_60.iloc[idx-1]
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
            })

        # ================= PREDICTIONS =================
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
                np.hstack([[pred_scaled], np.zeros(len(features)-1)]).reshape(1, -1)
            )[0][0]

            # Stability controls (MT5 logic)
            pred_ret = np.clip(pred_ret, -2.5*volatility, 2.5*volatility)
            pred_price = last_price * (1 + pred_ret)

            ema = temp_df.iloc[-1]["EMA20"]
            pred_price = 0.9 * pred_price + 0.1 * ema
            pred_price *= (1 + np.random.normal(0, volatility/2))

            future_time = temp_df.iloc[-1]["time"] + timedelta(minutes=1)

            # Realistic wicks
            body = abs(pred_price - last_price)
            wick = max(body * 0.5, last_price * volatility * 0.2)

            high_p = max(last_price, pred_price) + wick
            low_p = min(last_price, pred_price) - wick

            new_row = {
                "time": future_time,
                "open": last_price,
                "high": high_p,
                "low": low_p,
                "close": pred_price,
                "volume": temp_df.iloc[-1]["volume"]
            }

            temp_df = pd.concat([temp_df, pd.DataFrame([new_row])], ignore_index=True)

            # Recalculate indicators
            temp_df["EMA20"] = temp_df["close"].ewm(span=20, adjust=False).mean()
            temp_df["SMA50"] = temp_df["close"].rolling(50).mean()
            temp_df["RSI"] = ta.momentum.RSIIndicator(temp_df["close"], window=14).rsi()
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
            })

            last_price = pred_price
            last_seq = scaler.transform(temp_df[features].tail(LOOKBACK))

        print("Sent 60 real + 10 future")

    except Exception as e:
        print("Error:", e)

    time.sleep(5)
