from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA = []

@app.get("/")
def root():
    return {"status": "backend running"}

@app.get("/data")
def get_data():
    return DATA

@app.post("/update")
def update(payload: dict):
    global DATA

    if payload["type"] == "prediction":
        # remove ALL old prediction candles
        DATA = [d for d in DATA if d.get("type") != "prediction"]

    DATA.extend(payload["candles"])

    # keep last 1000 real candles only
    real = [d for d in DATA if d["type"] == "real"][-1000:]
    pred = [d for d in DATA if d["type"] == "prediction"]

    DATA = real + pred
    return {"status": "ok"}
