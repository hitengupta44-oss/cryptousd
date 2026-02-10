from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

REAL_DATA = []
PRED_DATA = []

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/update")
def update(payload: dict):
    global REAL_DATA, PRED_DATA

    if payload["type"] == "real":
        # Add real candle
        REAL_DATA.append(payload)

        # Keep only last 60 real candles
        REAL_DATA = REAL_DATA[-60:]

        # Clear old predictions (important)
        PRED_DATA = []

    elif payload["type"] == "prediction":
        PRED_DATA.append(payload)

        # Keep only last 10 predictions
        PRED_DATA = PRED_DATA[-10:]

    return {"status": "ok"}

@app.get("/data")
def get_data():
    return REAL_DATA + PRED_DATA
