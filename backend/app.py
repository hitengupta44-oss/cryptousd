from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# in-memory store (simple + safe)
latest_data = {
    "time": None,
    "price": None,
    "prediction": None
}

@app.get("/")
def root():
    return {"status": "Backend running"}

@app.post("/update")
def update(data: dict):
    global latest_data
    latest_data = data
    return {"status": "updated"}

@app.get("/latest")
def latest():
    return latest_data
