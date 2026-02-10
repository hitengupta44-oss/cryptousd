from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

DATA = []
MAX_REAL = 120  # keep history


@app.get("/")
def home():
    return {"status": "running"}


@app.post("/update")
def update(payload: dict):
    global DATA

    payload_type = payload.get("type", "unknown")

    # If real candle arrives → remove old predictions only
    if payload_type == "real":
        DATA = [d for d in DATA if d.get("type") != "prediction"]

    DATA.append(payload)

    # Keep only last real candles
    real = [d for d in DATA if d.get("type") == "real"]
    pred = [d for d in DATA if d.get("type") == "prediction"]

    if len(real) > MAX_REAL:
        real = real[-MAX_REAL:]

    DATA = real + pred

    return {"status": "ok"}


@app.get("/data")
def get_data():
    return DATA
