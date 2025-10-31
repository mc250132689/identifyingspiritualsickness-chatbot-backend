from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from langdetect import detect
import difflib
import json
import os
import requests

# === FastAPI setup ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Hugging Face GPT-OSS client ===
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(provider="groq", api_key=HF_TOKEN)

# === Training data file ===
DATA_FILE = "training_data.json"
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump({"training_data": []}, f, ensure_ascii=False, indent=2)

# === Models ===
class ChatRequest(BaseModel):
    message: str


class TrainRequest(BaseModel):
    question: str
    answer: str

# === Utility functions ===
def load_data():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)["training_data"]

def save_data(data_list):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump({"training_data": data_list}, f, ensure_ascii=False, indent=2)

def translate(text, source, target):
    """Translate text using Hugging Face translation models"""
    model_map = {
        ("en", "ms"): "Helsinki-NLP/opus-mt-en-ms",
        ("ms", "en"): "Helsinki-NLP/opus-mt-ms-en",
        ("en", "ar"): "Helsinki-NLP/opus-mt-en-ar",
        ("ar", "en"): "Helsinki-NLP/opus-mt-ar-en",
        ("ms", "ar"): "Helsinki-NLP/opus-mt-ms-ar",
        ("ar", "ms"): "Helsinki-NLP/opus-mt-ar-ms"
    }
    if (source, target) not in model_map:
        return text
    model = model_map[(source, target)]
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    try:
        r = requests.post(
            f"https://api-inference.huggingface.co/models/{model}",
            headers=headers,
            json={"inputs": text},
            timeout=60,
        )
        return r.json()[0]["translation_text"]
    except Exception:
        return text

# === Chat endpoint (updated to log all user messages) ===
@app.post("/chat")
async def chat(req: ChatRequest):
    user_message = req.message.strip()
    if not user_message:
        return {"response": "Please type a message."}

    # Detect language
    try:
        lang = detect(user_message)
    except Exception:
        lang = "en"

    # Load training data
    data = load_data()

    # 1️⃣ Check training data for close matches in the same language
    questions_in_lang = [item["question"] for item in data if item["lang"] == lang]
    if questions_in_lang:
        match = difflib.get_close_matches(user_message, questions_in_lang, n=1, cutoff=0.6)
        if match:
            for item in data:
                if item["lang"] == lang and item["question"] == match[0]:
                    # Log the message even if answer exists
                    data.append({"question": user_message, "answer": item["answer"], "lang": lang})
                    save_data(data)
                    return {"response": item["answer"]}

    # 2️⃣ No match → ask GPT-OSS
    eng_msg = user_message if lang == "en" else translate(user_message, lang, "en")

    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an Islamic assistant specializing in identifying spiritual sickness, "
                    "sihr, jinn disturbances, and providing guidance according to Quran, Sunnah, and authentic ruqyah. "
                    "Keep your answers strictly Islamic and practical."
                ),
            },
            {"role": "user", "content": eng_msg},
        ],
    )

    reply = completion.choices[0].message["content"]

    if lang != "en":
        reply = translate(reply, "en", lang)

    # ✅ Log every user message for later review
    data.append({"question": user_message, "answer": reply, "lang": lang})
    save_data(data)

    return {"response": reply}

# === Train endpoint ===
@app.post("/train")
async def train(req: TrainRequest):
    question = req.question.strip()
    answer = req.answer.strip()
    if not question or not answer:
        return {"message": "Please provide both question and answer."}

    try:
        lang = detect(question)
    except Exception:
        lang = "en"

    data = load_data()
    # Update existing if duplicate
    for item in data:
        if item["lang"] == lang and item["question"].lower() == question.lower():
            item["answer"] = answer
            save_data(data)
            return {"message": f"Updated existing {lang.upper()} training data."}

    # Add new
    data.append({"question": question, "answer": answer, "lang": lang})
    save_data(data)
    return {"message": f"Added new {lang.upper()} training data successfully."}
