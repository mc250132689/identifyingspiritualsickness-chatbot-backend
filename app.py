from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from langdetect import detect
import difflib
import json
import os

# ------------------ CONFIG ------------------
DATA_FILE = "training_data.json"
ADMIN_KEY = os.getenv("ADMIN_KEY", "mc250132689")

# HuggingFace Model Client (GPT-OSS 20B)
client = InferenceClient(
    provider="groq",
    api_key=os.getenv("HF_TOKEN")
)

app = FastAPI()

# Enable CORS for your frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # You can restrict to your UI domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ UTILITIES ------------------

def load_data():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

trained_answers = {}

def load_memory():
    data = load_data()
    for item in data:
        lang = item.get("lang", "en")
        q = item["question"].lower()
        if lang not in trained_answers:
            trained_answers[lang] = {}
        trained_answers[lang][q] = item["answer"]

load_memory()

# Islamic Rule Enforcement
ISLAMIC_RULES = """
You are an Islamic assistant specializing in:
- Spiritual sickness (jinn disturbance, sihr/black magic, evil eye)
- Islamic medical practices & ruqyah from Qur'an & Sunnah

RULES YOU MUST FOLLOW:
1. Your guidance must follow Quran, authentic Sunnah, scholars consensus.
2. Do NOT mention any non-Islamic spiritual healing or witchcraft.
3. Do NOT recommend crystals, tarot, zodiac, reiki, yoga chakra, energy cleansing, etc.
4. Do NOT claim to diagnose possession with certainty. Use conditional wording.
5. Always encourage patience, dua, dhikr, ruqyah, salah, good character, Quran recitation.
6. If user describes severe symptoms → recommend they consult a qualified ruqyah practitioner.
"""

# ------------------ REQUEST MODEL ------------------

class ChatRequest(BaseModel):
    message: str

# ------------------ CHATBOT ENDPOINT ------------------

@app.post("/chat")
async def chat(req: ChatRequest):
    user_message = req.message.strip()
    if not user_message:
        return {"response": "Please type a message."}

    try:
        lang = detect(user_message)
    except:
        lang = "en"

    # Detect known symptoms
    symptoms = []
    symptom_keywords = {
        "nightmare": "Recurring bad dreams",
        "sleep paralysis": "Sleep paralysis episodes",
        "shadow": "Seeing black shadows",
        "jinn": "Possible jinn disturbance",
        "magic": "Possible sihr / black magic",
        "sihr": "Signs of sihr (black magic)",
        "ruqyah": "Seeking ruqyah guidance",
        "waswas": "Waswas (whispering doubts)"
    }

    for key, label in symptom_keywords.items():
        if key in user_message.lower():
            symptoms.append(label)

    if symptoms:
        formatted = "\n- ".join(symptoms)
        return {"response":
            f"🕌 *Possible Spiritual Symptoms Noticed*\n\n"
            f"You mentioned signs related to:\n- {formatted}\n\n"
            f"Recommended steps:\n"
            f"1. Recite Surah Al-Baqarah daily\n"
            f"2. Recite Ayat al-Kursi before sleep\n"
            f"3. Play Ruqyah audio (Mishary Rashid / Saad Al-Ghamdi)\n"
            f"4. Maintain wudu and reduce stress\n\n"
            f"If symptoms persist or intensify, consult a *qualified ruqyah practitioner*."
        }

    # Check learned answers
    lang_dict = trained_answers.get(lang, {})
    match = difflib.get_close_matches(user_message.lower(), lang_dict.keys(), n=1, cutoff=0.6)
    if match:
        return {"response": lang_dict[match[0]]}

    # Generate response using GPT-OSS 20B
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": ISLAMIC_RULES},
            {"role": "user", "content": user_message},
        ],
    )

    reply = completion.choices[0].message["content"]

    # Save new learned answer to dataset
    data = load_data()
    data.append({"question": user_message, "answer": reply, "lang": lang})
    save_data(data)
    trained_answers.setdefault(lang, {})[user_message.lower()] = reply

    return {"response": reply}

# ------------------ TRAINING VIEW ENDPOINTS ------------------

@app.get("/get-training-data")
async def get_training_data():
    return {"training_data": load_data()}

@app.get("/admin-stats")
async def admin_stats(key: str):
    if key != ADMIN_KEY:
        return {"error": "Unauthorized"}

    data = load_data()
    if not data:
        return {"total_records": 0, "avg_question_length": 0, "avg_answer_length": 0}

    avg_q = sum(len(d["question"]) for d in data) // len(data)
    avg_a = sum(len(d["answer"]) for d in data) // len(data)

    return {
        "total_records": len(data),
        "avg_question_length": avg_q,
        "avg_answer_length": avg_a,
    }
