from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend access (Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= STORAGE =================
REAL_DATA = []
PRED_DATA = []

MAX_REAL = 60
MAX_PRED = 10


@app.get("/")
def home():
    return {"status": "running"}


# ================= UPDATE ENDPOINT =================
@app.post("/update")
def update(payload: dict):
    global REAL_DATA, PRED_DATA

    data_type = payload.get("type")

    if data_type == "real":
        # Remove old predictions when new real candle comes
        PRED_DATA = []

        # Avoid duplicate real candle
        if not REAL_DATA or REAL_DATA[-1]["time"] != payload["time"]:
            REAL_DATA.append(payload)

        # Keep only last 60 real candles
        if len(REAL_DATA) > MAX_REAL:
            REAL_DATA = REAL_DATA[-MAX_REAL:]

    elif data_type == "prediction":
        # Avoid duplicate prediction timestamps
        if not PRED_DATA or PRED_DATA[-1]["time"] != payload["time"]:
            PRED_DATA.append(payload)

        # Keep only last 10 predictions
        if len(PRED_DATA) > MAX_PRED:
            PRED_DATA = PRED_DATA[-MAX_PRED:]

    return {"status": "ok"}


# ================= DATA ENDPOINT =================
@app.get("/data")
def get_data():
    # Always return real + prediction together
    return REAL_DATA + PRED_DATA
