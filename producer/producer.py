import requests
import time
import random
from datetime import datetime

BACKEND_URL = "https://YOUR-BACKEND.onrender.com/update"

price = 43000

while True:
    price += random.uniform(-50, 50)
    prediction = price + random.uniform(-100, 100)

    payload = {
        "time": datetime.utcnow().isoformat(),
        "price": round(price, 2),
        "prediction": round(prediction, 2)
    }

    try:
        requests.post(BACKEND_URL, json=payload, timeout=5)
        print("Sent:", payload)
    except Exception as e:
        print("Error:", e)

    time.sleep(60)
