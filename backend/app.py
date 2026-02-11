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
return {"status": "backend running"}

@app.post("/update")
def update(payload: dict):
global REAL_DATA, PRED_DATA, last_real_time

```
t = payload["time"]
typ = payload.get("type")

# ===== REAL DATA =====
if typ == "real":
    REAL_DATA[t] = payload

    # Clear predictions only when a NEW candle arrives
    if last_real_time is None or t > last_real_time:
        last_real_time = t
        PRED_DATA = {}

    # Keep only last 60
    if len(REAL_DATA) > MAX_REAL:
        for k in sorted(REAL_DATA.keys())[:-MAX_REAL]:
            del REAL_DATA[k]

# ===== PREDICTIONS =====
elif typ == "prediction":
    PRED_DATA[t] = payload

    if len(PRED_DATA) > MAX_PRED:
        for k in sorted(PRED_DATA.keys())[:-MAX_PRED]:
            del PRED_DATA[k]

return {"status": "ok"}
```

@app.get("/data")
def get_data():
real = [REAL_DATA[k] for k in sorted(REAL_DATA.keys())]
pred = [PRED_DATA[k] for k in sorted(PRED_DATA.keys())]
return real + pred
