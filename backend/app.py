from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

REAL_DATA = []
PRED_DATA = []

MAX_REAL = 60


@app.get("/")
def home():
    return {"status": "running"}


@app.post("/update")
def update(payload: dict):
    global REAL_DATA, PRED_DATA

    if "type" not in payload:
        return {"status": "ignored"}

    # ================= REAL CANDLES =================
    if payload["type"] == "real":
        # Remove duplicate timestamps
        REAL_DATA = [d for d in REAL_DATA if d["time"] != payload["time"]]

        REAL_DATA.append(payload)

        # Keep only last 60 real candles
        REAL_DATA = sorted(REAL_DATA, key=lambda x: x["time"])[-MAX_REAL:]

        # When a new real candle arrives → clear old predictions
        PRED_DATA = []

    # ================= PREDICTIONS =================
    elif payload["type"] == "prediction":
        # Avoid duplicate prediction times
        PRED_DATA = [d for d in PRED_DATA if d["time"] != payload["time"]]
        PRED_DATA.append(payload)

        # Keep only next 10
        PRED_DATA = sorted(PRED_DATA, key=lambda x: x["time"])[:10]

    return {"status": "ok"}


@app.get("/data")
def get_data():
    # Return real + predictions combined
    combined = sorted(REAL_DATA + PRED_DATA, key=lambda x: x["time"])
    return combined
