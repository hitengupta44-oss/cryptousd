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

data_store = []

@app.post("/update")
def update(data: dict):
    data["timestamp"] = datetime.utcnow().isoformat()
    data_store.append(data)

    if len(data_store) > 500:
        data_store.pop(0)

    return {"status": "ok"}

@app.get("/history")
def history():
    return data_store
