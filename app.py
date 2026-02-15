from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests
import google.generativeai as genai
import os

# =====================================
# CONFIGURATION
# =====================================

API_KEY = os.getenv("GEMINI_API_KEY")
print("Gemini key loaded:", API_KEY is not None)

if not API_KEY:
    print("WARNING: GEMINI_API_KEY not set!")

genai.configure(api_key=API_KEY)

LIVE_API = "https://cryptoliveusdt.onrender.com/data/"

app = FastAPI()

# =====================================
# CORS (for your HTML UI)
# =====================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================
# AUTO-DETECT BEST GEMINI MODEL
# =====================================

def load_best_model():
    preferred = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-1.5-flash",
        "gemini-1.5-pro"
    ]

    try:
        available = [
            m.name.replace("models/", "")
            for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        ]

        print("Available models:", available)

        for p in preferred:
            if p in available:
                print("Using model:", p)
                return genai.GenerativeModel(p)

        # fallback to first available
        if available:
            print("Fallback model:", available[0])
            return genai.GenerativeModel(available[0])

    except Exception as e:
        print("Model detection error:", e)

    return None


model = load_best_model()

# =====================================
# REQUEST MODEL
# =====================================

class ChatRequest(BaseModel):
    message: str


# =====================================
# TRADING DETECTION
# =====================================

TRADING_KEYWORDS = [
    "btc", "bitcoin", "price", "market", "trade",
    "trading", "buy", "sell", "long", "short",
    "rsi", "ema", "sma", "indicator",
    "trend", "signal", "crypto", "candles"
]

def is_trading_query(message: str):
    msg = message.lower()
    return any(word in msg for word in TRADING_KEYWORDS)


# =====================================
# FETCH LIVE MARKET
# =====================================

def get_market_summary():
    try:
        res = requests.get(LIVE_API, timeout=6)
        data = res.json()

        real_data = [c for c in data if c.get("type") == "real"]
        latest = real_data[-1] if real_data else data[-1]

        pred_data = [c for c in data if c.get("type") == "prediction"]
        next_pred = pred_data[0]["close"] if pred_data else None

        return {
            "price": latest.get("close"),
            "ema20": latest.get("ema20"),
            "sma50": latest.get("sma50"),
            "rsi": latest.get("rsi"),
            "signal": latest.get("signal"),
            "prediction": next_pred
        }

    except Exception as e:
        print("Market API error:", e)
        return None


# =====================================
# GEMINI SAFE CALL
# =====================================

def ask_gemini(prompt):
    if not model:
        return "AI model not available."

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print("Gemini error:", e)
        return "AI service temporarily unavailable."


# =====================================
# CHAT ENDPOINT
# =====================================

@app.post("/chat")
def chat(req: ChatRequest):
    user_message = req.message

    general_reply = ask_gemini(user_message)

    if is_trading_query(user_message):
        market = get_market_summary()

        if market:
            market_prompt = f"""
You are a professional BTCUSD trading analyst.

Live Market:
Price: {market['price']}
EMA20: {market['ema20']}
SMA50: {market['sma50']}
RSI: {market['rsi']}
Signal: {market['signal']}
Next prediction: {market['prediction']}

User question: {user_message}

Give:
- Trend direction
- Momentum
- Short-term outlook
Keep it concise.
"""
            market_reply = ask_gemini(market_prompt)
        else:
            market_reply = "Live market data unavailable."

        final_reply = f"""
📊 Live Market Insight:
{market_reply}

📘 General Explanation:
{general_reply}
"""
    else:
        final_reply = general_reply

    return {"reply": final_reply}


# =====================================
# HEALTH & TEST ENDPOINTS
# =====================================

@app.get("/")
def home():
    return {"status": "BTC Gemini Chatbot Running"}

@app.get("/test-gemini")
def test_gemini():
    return {"reply": ask_gemini("Say hello in one sentence.")}

@app.get("/test-market")
def test_market():
    return {"market": get_market_summary()}
