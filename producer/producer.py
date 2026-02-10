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

# SETTINGS
SYMBOL = "BTCUSD"
TIMEFRAME = mt5.TIMEFRAME_M1
LOOKBACK = 60
PRED_MINUTES = 10
ROLLING_CANDLES = 500
BACKEND_URL = "https://cryptousdlive-1.onrender.com/update"

# INIT MT5
if not mt5.initialize():
    raise RuntimeError("MT5 init failed")

# MODEL
features = ['close','EMA20','SMA50','RSI','MACD','MACD_Signal','VWAP']
scaler = MinMaxScaler()

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(LOOKBACK,len(features))),
    Dropout(0.2),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

last_trained = None


def add_indicators(df):
    df['EMA20'] = df['close'].ewm(span=20).mean()
    df['SMA50'] = df['close'].rolling(50).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['close']).rsi()

    macd = ta.trend.MACD(df['close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()

    df['VWAP'] = (df['close'] * df['tick_volume']).cumsum() / df['tick_volume'].cumsum()

    return df.dropna()


def create_sequences(data):
    X,y=[],[]
    for i in range(LOOKBACK,len(data)):
        X.append(data[i-LOOKBACK:i])
        y.append(data[i,0])
    return np.array(X), np.array(y)


print("Producer started")

while True:
    try:
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, ROLLING_CANDLES)
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        df = add_indicators(df)

        scaled = scaler.fit_transform(df[features])
        X,y = create_sequences(scaled)

        # retrain every 30 min
        now = datetime.utcnow()
        if last_trained is None or (now-last_trained).seconds > 1800:
            model.fit(X,y,epochs=3,batch_size=32,verbose=0)
            last_trained = now
            print("Model updated")

        # ===== REAL CANDLE =====
        last = df.iloc[-1]

        signal = None
        if df['EMA20'].iloc[-1] > df['SMA50'].iloc[-1] and df['EMA20'].iloc[-2] <= df['SMA50'].iloc[-2]:
            signal = "BUY"
        elif df['EMA20'].iloc[-1] < df['SMA50'].iloc[-1] and df['EMA20'].iloc[-2] >= df['SMA50'].iloc[-2]:
            signal = "SELL"

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
            "signal": signal,
            "type": "real"
        }

        requests.post(BACKEND_URL,json=real_payload)

        # ===== FUTURE PREDICTION =====
        last_seq = scaled[-LOOKBACK:].copy()
        temp_df = df.copy()
        base_time = last["time"]

        for i in range(PRED_MINUTES):
            pred_scaled = model.predict(last_seq.reshape(1,LOOKBACK,len(features)),verbose=0)[0][0]

            pred_close = scaler.inverse_transform(
                np.hstack([[pred_scaled],np.zeros(len(features)-1)]).reshape(1,-1)
            )[0][0]

            new_time = base_time + timedelta(minutes=i+1)
            prev_close = temp_df.iloc[-1]['close']

            new_row = {
                "time": new_time,
                "open": prev_close,
                "high": max(prev_close,pred_close),
                "low": min(prev_close,pred_close),
                "close": pred_close,
                "tick_volume": temp_df.iloc[-1]['tick_volume']
            }

            temp_df = pd.concat([temp_df,pd.DataFrame([new_row])],ignore_index=True)
            temp_df = add_indicators(temp_df)

            last_seq = scaler.transform(temp_df[features].tail(LOOKBACK))

            payload = {
                "time": new_time.isoformat(),
                "open": float(prev_close),
                "high": float(new_row["high"]),
                "low": float(new_row["low"]),
                "close": float(pred_close),
                "type": "prediction"
            }

            requests.post(BACKEND_URL,json=payload)

        print("Updated at", last["time"])

    except Exception as e:
        print("Error:", e)

    time.sleep(60)
