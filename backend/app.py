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
MAX_RECORDS = 2000


@app.get("/")
def home():
    return {"status": "running"}


@app.post("/update")
def update(payload: dict):
    global DATA

    # Ensure type exists
    if "type" not in payload:
        payload["type"] = "unknown"

    # When a real candle arrives → remove old predictions
    if payload["type"] == "real":
        DATA = [d for d in DATA if d.get("type") != "prediction"]

    DATA.append(payload)

    # Keep only last records
    if len(DATA) > MAX_RECORDS:
        DATA = DATA[-MAX_RECORDS:]

    return {"status": "ok"}


@app.get("/data")
def get_data():
    return DATA
