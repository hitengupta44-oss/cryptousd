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
SYMBOL = "BTCUSD"
TIMEFRAME = mt5.TIMEFRAME_M1
LOOKBACK = 60
FUTURE_MINUTES = 10
BACKEND_URL = "https://cryptousd.onrender.com/update"

# ================= MT5 ======================
if not mt5.initialize():
    raise RuntimeError("MT5 init failed")

# ================= MODEL ====================
def build_model(n_features):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LOOKBACK, n_features)),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

model = None
scaler = MinMaxScaler()

# ================= LOOP =====================
while True:
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 500)
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    # ===== INDICATORS =====
    df["EMA20"] = df["close"].ewm(span=20).mean()
    df["SMA50"] = df["close"].rolling(50).mean()
    df["RSI"] = ta.momentum.RSIIndicator(df["close"]).rsi()

    macd = ta.trend.MACD(df["close"])
    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()

    df["VWAP"] = (df["close"] * df["tick_volume"]).cumsum() / df["tick_volume"].cumsum()

    df.dropna(inplace=True)

    features = ["close", "EMA20", "SMA50", "RSI", "MACD", "MACD_SIGNAL", "VWAP"]
    scaled = scaler.fit_transform(df[features])

    X, y = [], []
    for i in range(LOOKBACK, len(scaled)):
        X.append(scaled[i-LOOKBACK:i])
        y.append(scaled[i, 0])

    X, y = np.array(X), np.array(y)

    if model is None:
        model = build_model(len(features))
        model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    # ===== 10-MIN FUTURE PREDICTION =====
    last_seq = X[-1].reshape(1, LOOKBACK, len(features))
    pred_scaled = model.predict(last_seq, verbose=0)[0][0]

    pred_price = scaler.inverse_transform(
        np.hstack([[pred_scaled], np.zeros(len(features)-1)]).reshape(1, -1)
    )[0][0]

    last = df.iloc[-1]

    # ===== MULTI-INDICATOR VOTING =====
    votes = []

    # EMA / SMA
    votes.append("BUY" if last["EMA20"] > last["SMA50"] else "SELL")

    # RSI
    if last["RSI"] < 30:
        votes.append("BUY")
    elif last["RSI"] > 70:
        votes.append("SELL")

    # MACD crossover
    if last["MACD"] > last["MACD_SIGNAL"]:
        votes.append("BUY")
    else:
        votes.append("SELL")

    signal = max(set(votes), key=votes.count)

    payload = {
        "time": (last["time"] + timedelta(minutes=FUTURE_MINUTES)).isoformat(),
        "open": float(last["close"]),
        "high": float(max(last["close"], pred_price)),
        "low": float(min(last["close"], pred_price)),
        "close": float(pred_price),

        "ema20": float(last["EMA20"]),
        "sma50": float(last["SMA50"]),
        "vwap": float(last["VWAP"]),
        "rsi": float(last["RSI"]),
        "macd": float(last["MACD"]),
        "macd_signal": float(last["MACD_SIGNAL"]),

        "signal": signal
    }

    requests.post(BACKEND_URL, json=payload, timeout=5)
    print("Sent:", payload)

    time.sleep(60)
