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

    if payload["type"] == "real":
        # Remove old predictions
        DATA = [d for d in DATA if d["type"] == "real"]

        DATA.append(payload)

        # Keep only last 60 real candles
        DATA = DATA[-MAX_REAL:]

    elif payload["type"] == "prediction":
        DATA.append(payload)

    return {"status": "ok"}


@app.get("/data")
def get_data():
    return DATA
