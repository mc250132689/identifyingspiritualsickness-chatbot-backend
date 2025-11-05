from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from langdetect import detect
import difflib
import os
import requests
from collections import Counter
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base

# === Database setup ===
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class TrainingData(Base):
    __tablename__ = "training_data"
    id = Column(Integer, primary_key=True, index=True)
    question = Column(String)
    answer = Column(String)
    lang = Column(String)

Base.metadata.create_all(bind=engine)

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

# === In-memory training dictionary ===
trained_answers = {}  # Structure: trained_answers[lang][question_lower] = answer

def load_data():
    """Load all training data from DB into in-memory dictionary"""
    db = SessionLocal()
    data_list = db.query(TrainingData).all()
    db.close()

    global trained_answers
    trained_answers = {}
    for item in data_list:
        if item.lang not in trained_answers:
            trained_answers[item.lang] = {}
        trained_answers[item.lang][item.question.lower()] = item.answer

    return data_list

# Initial load
load_data()

# === Request models ===
class ChatRequest(BaseModel):
    message: str

class TrainRequest(BaseModel):
    question: str
    answer: str

# === Utilities ===
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

def detect_spiritual_symptoms(text):
    """Check user message for known spiritual symptoms"""
    text_lower = text.lower()
    symptoms = {
        "nightmares": "Frequent nightmares",
        "hearing whispers": "Hearing whispers",
        "sleep paralysis": "Sleep paralysis",
        "sudden anger": "Sudden intense anger",
        "fear of quran": "Discomfort when hearing Quran",
        "pressure on chest": "Chest tightness when sleeping"
    }
    detected = [sym for key, sym in symptoms.items() if key in text_lower]
    return detected

# === Endpoints ===

@app.post("/chat")
async def chat(req: ChatRequest):
    user_message = req.message.strip()
    if not user_message:
        return {"response": "Please type a message."}

    try:
        lang = detect(user_message)
    except Exception:
        lang = "en"

    # Check spiritual symptoms first
    symptoms_found = detect_spiritual_symptoms(user_message)
    if symptoms_found:
        return {"response":
            "🕌 *Possible Spiritual Disturbance Noticed*\n\n"
            "Based on your symptoms:\n- " + "\n- ".join(symptoms_found) +
            "\n\nRecommended actions:\n"
            "1. Perform Surah Al-Baqarah daily\n"
            "2. Recite Ayat al-Kursi before sleeping\n"
            "3. Play Ruqyah audio (Mishary Rashid)\n"
            "4. Avoid sin, maintain wudu, reduce stress\n\n"
            "If symptoms intensify, refer to a **qualified ruqyah practitioner**."
        }

    # 1️⃣ Check in-memory trained answers
    lang_dict = trained_answers.get(lang, {})
    match = difflib.get_close_matches(user_message.lower(), lang_dict.keys(), n=1, cutoff=0.6)
    if match:
        return {"response": lang_dict[match[0]]}

    # 2️⃣ Query GPT-OSS via Hugging Face
    eng_msg = user_message if lang == "en" else translate(user_message, lang, "en")
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": (
                "You are an Islamic assistant specializing in identifying spiritual sickness, "
                "sihr, jinn disturbances, and providing guidance according to Quran, Sunnah, and authentic ruqyah. "
                "Keep your answers strictly Islamic and practical."
            )},
            {"role": "user", "content": eng_msg},
        ]
    )
    reply = completion.choices[0].message["content"]

    if lang != "en":
        reply = translate(reply, "en", lang)

    # Save to DB and update in-memory dict
    db = SessionLocal()
    existing = db.query(TrainingData).filter(
        TrainingData.question.ilike(user_message),
        TrainingData.lang == lang
    ).first()
    if existing:
        existing.answer = reply
    else:
        db.add(TrainingData(question=user_message, answer=reply, lang=lang))
    db.commit()
    db.close()

    load_data()  # refresh in-memory dict

    return {"response": reply}

@app.post("/train")
async def train(req: TrainRequest):
    question = req.question.strip()
    answer = req.answer.strip()

    try:
        lang = detect(question)
    except Exception:
        lang = "en"

    db = SessionLocal()
    existing = db.query(TrainingData).filter(
        TrainingData.question.ilike(question),
        TrainingData.lang == lang
    ).first()
    if existing:
        existing.answer = answer
        msg = "Updated existing response."
    else:
        db.add(TrainingData(question=question, answer=answer, lang=lang))
        msg = "Added new response."
    db.commit()
    db.close()

    load_data()  # refresh in-memory dict
    return {"message": msg}

@app.get("/get-training-data")
async def get_training_data():
    data = load_data()
    return {"training_data": [
        {"question": d.question, "answer": d.answer, "lang": d.lang} for d in data
    ]}

# === Admin analytics ===
ADMIN_KEY = os.getenv("ADMIN_KEY", "mc250132689")

@app.get("/admin-stats")
async def admin_stats(key: str = Query(...)):
    if key != ADMIN_KEY:
        return {"error": "Unauthorized"}

    data = load_data()
    total_records = len(data)
    lang_count = Counter(item.lang for item in data)
    avg_q_len = round(sum(len(item.question) for item in data)/total_records,1) if total_records else 0
    avg_a_len = round(sum(len(item.answer) for item in data)/total_records,1) if total_records else 0

    return {
        "total_records": total_records,
        "languages": dict(lang_count),
        "avg_question_length": avg_q_len,
        "avg_answer_length": avg_a_len
    }
