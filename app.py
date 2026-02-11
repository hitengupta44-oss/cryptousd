from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

REAL_DATA = {}
PRED_DATA = {}

MAX_REAL = 60
MAX_PRED = 10


@app.get("/")
def home():
    return {"status": "running"}


@app.post("/update")
def update(payload: dict):
    global REAL_DATA, PRED_DATA

    t = payload["time"]

    if payload.get("type") == "real":
        # Replace candle if same timestamp
        REAL_DATA[t] = payload

        # Keep only latest 60
        if len(REAL_DATA) > MAX_REAL:
            sorted_keys = sorted(REAL_DATA.keys())
            for k in sorted_keys[:-MAX_REAL]:
                del REAL_DATA[k]

        # New real candle → clear predictions
        PRED_DATA = {}

    elif payload.get("type") == "prediction":
        PRED_DATA[t] = payload

        if len(PRED_DATA) > MAX_PRED:
            sorted_keys = sorted(PRED_DATA.keys())
            for k in sorted_keys[:-MAX_PRED]:
                del PRED_DATA[k]

    return {"status": "ok"}


@app.get("/data")
def get_data():
    real = [REAL_DATA[k] for k in sorted(REAL_DATA.keys())]
    pred = [PRED_DATA[k] for k in sorted(PRED_DATA.keys())]
    return real + pred
