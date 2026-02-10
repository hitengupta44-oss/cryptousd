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
MAX_REAL = 60

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/update")
def update(payload: dict):
    global DATA

    payload_type = payload.get("type", "unknown")

    # When real candle arrives
    if payload_type == "real":
        # Remove old predictions
        DATA = [d for d in DATA if d.get("type") != "prediction"]

        # Add real
        DATA.append(payload)

        # Keep last 60 real candles
        real = [d for d in DATA if d.get("type") == "real"]
        real = real[-MAX_REAL:]

        DATA = real

    elif payload_type == "prediction":
        DATA.append(payload)

    return {"status": "ok"}

@app.get("/data")
def get_data():
    return DATA
