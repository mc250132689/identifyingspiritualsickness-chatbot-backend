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

# === In-memory training dictionary ===
# Structure: trained_answers[lang][question_lower] = answer
trained_answers = {}

def load_data():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data_list = json.load(f)["training_data"]
    # Populate in-memory dictionary
    global trained_answers
    trained_answers = {}
    for item in data_list:
        lang = item["lang"]
        q = item["question"].lower()
        a = item["answer"]
        if lang not in trained_answers:
            trained_answers[lang] = {}
        trained_answers[lang][q] = a
    return data_list

def save_data(data_list):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump({"training_data": data_list}, f, ensure_ascii=False, indent=2)

# Initial load
load_data()

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

# === Request models ===
class ChatRequest(BaseModel):
    message: str

class TrainRequest(BaseModel):
    question: str
    answer: str

# === Chat endpoint ===
@app.post("/chat")
async def chat(req: ChatRequest):
    user_message = req.message.strip()
    if not user_message:
        return {"response": "Please type a message."}

    try:
        lang = detect(user_message)
    except Exception:
        lang = "en"

    # 1️⃣ Check in-memory trained answers first
    lang_dict = trained_answers.get(lang, {})
    match = difflib.get_close_matches(user_message.lower(), lang_dict.keys(), n=1, cutoff=0.6)
    if match:
        answer = lang_dict[match[0]]
        return {"response": answer}

    # 2️⃣ No match → query GPT-OSS via Hugging Face InferenceClient
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

    # Translate back if needed
    if lang != "en":
        reply = translate(reply, "en", lang)

    # Save to JSON and in-memory
    data = load_data()
    data.append({"question": user_message, "answer": reply, "lang": lang})
    save_data(data)
    if lang not in trained_answers:
        trained_answers[lang] = {}
    trained_answers[lang][user_message.lower()] = reply

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
    updated = False
    for item in data:
        if item["lang"] == lang and item["question"].lower() == question.lower():
            item["answer"] = answer
            updated = True
            break
    if not updated:
        data.append({"question": question, "answer": answer, "lang": lang})
    save_data(data)

    # Update in-memory trained_answers instantly
    if lang not in trained_answers:
        trained_answers[lang] = {}
    trained_answers[lang][question.lower()] = answer

    return {"message": f"{'Updated' if updated else 'Added'} {lang.upper()} training data successfully."}

# === Get all training data ===
@app.get("/get-training-data")
async def get_training_data():
    data = load_data()
    return {"training_data": data}
