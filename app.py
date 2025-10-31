from fastapi import FastAPI, Request
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

HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(provider="groq", api_key=HF_TOKEN)
DATA_FILE = "training_data.json"
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)

# === Models ===
class ChatRequest(BaseModel):
    message: str


class TrainRequest(BaseModel):
    question: str
    answer: str


# === Utility functions ===
def load_data():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def translate(text, source, target):
    """Translate between EN, MS, AR using HF translation models."""
    model_map = {
        ("en", "ms"): "Helsinki-NLP/opus-mt-en-ms",
        ("ms", "en"): "Helsinki-NLP/opus-mt-ms-en",
        ("en", "ar"): "Helsinki-NLP/opus-mt-en-ar",
        ("ar", "en"): "Helsinki-NLP/opus-mt-ar-en",
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


# === Core logic ===
@app.post("/chat")
async def chat(req: ChatRequest):
    user_message = req.message.strip()
    if not user_message:
        return {"response": "Please type a message."}

    try:
        lang = detect(user_message)
    except Exception:
        lang = "en"

    data = load_data()

    # 1️⃣ Check local training data for close matches
    questions = [item["question"] for item in data if item["lang"] == lang]
    if questions:
        match = difflib.get_close_matches(user_message, questions, n=1, cutoff=0.6)
        if match:
            for item in data:
                if item["lang"] == lang and item["question"] == match[0]:
                    return {"response": item["answer"]}

    # 2️⃣ If not found, ask GPT-OSS model
    eng_msg = user_message if lang == "en" else translate(user_message, lang, "en")

    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an Islamic assistant specializing in diagnosing and explaining "
                    "spiritual sicknesses (such as sihr, jinn, and ayn) based strictly on the Quran, "
                    "Sunnah, and authentic ruqyah practices. "
                    "Avoid Western psychology or superstition. Keep answers concise and practical."
                ),
            },
            {"role": "user", "content": eng_msg},
        ],
    )

    reply = completion.choices[0].message["content"]
    if lang != "en":
        reply = translate(reply, "en", lang)

    return {"response": reply}


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
    for item in data:
        if item["lang"] == lang and item["question"].lower() == question.lower():
            item["answer"] = answer
            save_data(data)
            return {"message": f"Updated existing {lang.upper()} data."}

    data.append({"question": question, "answer": answer, "lang": lang})
    save_data(data)
    return {"message": f"Added new {lang.upper()} training data successfully."}
