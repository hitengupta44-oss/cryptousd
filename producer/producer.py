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
BACKEND_URL = "https://cryptousd.onrender.com/update"

# ================= MT5 INIT =================
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

# ================= CORE FUNCTION =================
def get_payload():
    global model

    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 500)
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    # Indicators
    df["EMA20"] = df["close"].ewm(span=20).mean()
    df["SMA50"] = df["close"].rolling(50).mean()
    df["RSI"] = ta.momentum.RSIIndicator(df["close"]).rsi()
    macd = ta.trend.MACD(df["close"])
    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()

    df.dropna(inplace=True)

    features = ["close", "EMA20", "SMA50", "RSI", "MACD", "MACD_SIGNAL"]
    data = scaler.fit_transform(df[features])

    X, y = [], []
    for i in range(LOOKBACK, len(data)):
        X.append(data[i-LOOKBACK:i])
        y.append(data[i, 0])

    X, y = np.array(X), np.array(y)

    if model is None:
        model = build_model(len(features))
        model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    last_seq = X[-1].reshape(1, LOOKBACK, len(features))
    pred_scaled = model.predict(last_seq, verbose=0)[0][0]

    pred_price = scaler.inverse_transform(
        np.hstack([[pred_scaled], np.zeros(len(features)-1)]).reshape(1, -1)
    )[0][0]

    last = df.iloc[-1]

    signal = "HOLD"
    if last["EMA20"] > last["SMA50"]:
        signal = "BUY"
    elif last["EMA20"] < last["SMA50"]:
        signal = "SELL"

    return {
        "time": last["time"].isoformat(),
        "open": float(last["open"]),
        "high": float(last["high"]),
        "low": float(last["low"]),
        "close": float(last["close"]),
        "prediction": round(float(pred_price), 2),
        "signal": signal
    }

# ================= LOOP =================
while True:
    payload = get_payload()
    requests.post(BACKEND_URL, json=payload, timeout=5)
    print("Sent:", payload)
    time.sleep(60)
