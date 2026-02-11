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

last_real_time = None


@app.get("/")
def home():
    return {"status": "running"}


@app.post("/update")
def update(payload: dict):
    global REAL_DATA, PRED_DATA, last_real_time

    t = payload["time"]
    typ = payload.get("type")

    # ================= REAL =================
    if typ == "real":
        REAL_DATA[t] = payload

        # Detect NEW candle (not historical resend)
        if last_real_time is None or t > last_real_time:
            last_real_time = t
            PRED_DATA = {}  # clear predictions only once per new candle

        # Keep only latest 60
        if len(REAL_DATA) > MAX_REAL:
            sorted_keys = sorted(REAL_DATA.keys())
            for k in sorted_keys[:-MAX_REAL]:
                del REAL_DATA[k]

    # ================= PRED =================
    elif typ == "prediction":
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
