# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for testing; restrict in production
    allow_methods=["*"],
    allow_headers=["*"]
)

DATA = []

@app.post("/update")
def update(payload: dict):
    # Ensure each candle has type
    if "type" not in payload:
        payload["type"] = "unknown"
    DATA.append(payload)

    # Keep last 2000 candles
    if len(DATA) > 2000:
        DATA.pop(0)
    return {"status": "ok"}

@app.get("/data")
def get_data():
    return DATA
