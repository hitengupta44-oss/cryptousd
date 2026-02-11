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
MAX_PRED = 10


@app.get("/")
def home():
    return {"status": "running"}


@app.post("/update")
def update(payload: dict):
    global REAL_DATA, PRED_DATA

    t = payload.get("type")

    if t == "real":
        # keep last 60 real candles
        REAL_DATA.append(payload)
        REAL_DATA = REAL_DATA[-MAX_REAL:]

        # new real candle → remove old predictions
        PRED_DATA = []

    elif t == "prediction":
        PRED_DATA.append(payload)
        PRED_DATA = PRED_DATA[-MAX_PRED:]

    return {"status": "ok"}


@app.get("/data")
def get_data():
    # return past + future together
    return REAL_DATA + PRED_DATA
