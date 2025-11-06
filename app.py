from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from langdetect import detect
import difflib
import json
import os
from datetime import datetime

# ------------------ CONFIG ------------------
DATA_FILE = "training_data.json"
CHAT_LOG_FILE = "chat_logs.json"
FEEDBACK_FILE = "feedback.json"
ADMIN_KEY = os.getenv("ADMIN_KEY", "mc250132689")

client = InferenceClient(
    provider="groq",
    api_key=os.getenv("HF_TOKEN")
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ UTILITIES ------------------
def load_data(file):
    if not os.path.exists(file):
        return []
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)

def save_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

trained_answers = {}
def load_memory():
    data = load_data(DATA_FILE)
    for item in data:
        lang = item.get("lang", "en")
        q = item["question"].lower()
        if lang not in trained_answers:
            trained_answers[lang] = {}
        trained_answers[lang][q] = item["answer"]
load_memory()

# Islamic Rules
ISLAMIC_RULES = """
You are an Islamic assistant specializing in spiritual sickness and ruqyah.
Rules:
1. Follow Quran, Sunnah, and scholars consensus.
2. Do NOT mention non-Islamic spiritual healing.
3. Use conditional wording for jinn/possession.
4. Encourage patience, dua, dhikr, ruqyah, salah, Quran recitation.
"""

# ------------------ MODELS ------------------
class ChatRequest(BaseModel):
    message: str

class TrainRequest(BaseModel):
    question: str
    answer: str

class FeedbackRequest(BaseModel):
    text: str

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

    # Symptoms detection
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
        reply = (
            f"🕌 *Possible Spiritual Symptoms Noticed*\n\n"
            f"You mentioned signs related to:\n- {formatted}\n\n"
            f"Recommended steps:\n"
            f"1. Recite Surah Al-Baqarah daily\n"
            f"2. Recite Ayat al-Kursi before sleep\n"
            f"3. Play Ruqyah audio (Mishary Rashid / Saad Al-Ghamdi)\n"
            f"4. Maintain wudu and reduce stress\n\n"
            f"If symptoms persist or intensify, consult a *qualified ruqyah practitioner*."
        )
        # Save chat log
        chat_logs = load_data(CHAT_LOG_FILE)
        chat_logs.append({"user": user_message, "bot": reply, "time": str(datetime.now())})
        save_data(CHAT_LOG_FILE, chat_logs)
        return {"response": reply}

    # Check trained answers
    lang_dict = trained_answers.get(lang, {})
    match = difflib.get_close_matches(user_message.lower(), lang_dict.keys(), n=1, cutoff=0.6)
    if match:
        reply = lang_dict[match[0]]
        chat_logs = load_data(CHAT_LOG_FILE)
        chat_logs.append({"user": user_message, "bot": reply, "time": str(datetime.now())})
        save_data(CHAT_LOG_FILE, chat_logs)
        return {"response": reply}

    # Generate using GPT-OSS 20B
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": ISLAMIC_RULES},
            {"role": "user", "content": user_message},
        ],
    )
    reply = completion.choices[0].message["content"]

    # Save new trained answer
    data = load_data(DATA_FILE)
    data.append({"question": user_message, "answer": reply, "lang": lang})
    save_data(DATA_FILE, data)
    trained_answers.setdefault(lang, {})[user_message.lower()] = reply

    # Save chat log
    chat_logs = load_data(CHAT_LOG_FILE)
    chat_logs.append({"user": user_message, "bot": reply, "time": str(datetime.now())})
    save_data(CHAT_LOG_FILE, chat_logs)

    return {"response": reply}

# ------------------ TRAINING ENDPOINT ------------------
@app.post("/train")
async def train(req: TrainRequest):
    data = load_data(DATA_FILE)
    data.append({"question": req.question, "answer": req.answer, "lang": "en"})
    save_data(DATA_FILE, data)
    trained_answers.setdefault("en", {})[req.question.lower()] = req.answer
    return {"message": "Training data submitted successfully."}

@app.get("/get-training-data")
async def get_training_data():
    return {"training_data": load_data(DATA_FILE)}

@app.delete("/delete-entry")
async def delete_entry(question: str, key: str):
    if key != ADMIN_KEY:
        return {"error": "Unauthorized"}
    data = load_data(DATA_FILE)
    new_data = [d for d in data if d["question"].lower() != question.lower()]
    save_data(DATA_FILE, new_data)
    load_memory()
    return {"status": "success", "message": "Entry deleted"}

@app.get("/admin-stats")
async def admin_stats(key: str):
    if key != ADMIN_KEY:
        return {"error": "Unauthorized"}
    data = load_data(DATA_FILE)
    chat_logs = load_data(CHAT_LOG_FILE)
    if not data:
        return {"total_records": 0, "avg_question_length": 0, "avg_answer_length": 0, "total_chats": len(chat_logs)}
    avg_q = sum(len(d["question"]) for d in data) // len(data)
    avg_a = sum(len(d["answer"]) for d in data) // len(data)
    return {
        "total_records": len(data),
        "avg_question_length": avg_q,
        "avg_answer_length": avg_a,
        "total_chats": len(chat_logs)
    }

@app.get("/get-chat-logs")
async def get_chat_logs(key: str):
    if key != ADMIN_KEY:
        return {"error": "Unauthorized"}
    logs = load_data(CHAT_LOG_FILE)
    return {"logs": logs}

# ------------------ FEEDBACK ------------------
@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    data = load_data(FEEDBACK_FILE)
    data.append({"text": req.text, "time": str(datetime.now())})
    save_data(FEEDBACK_FILE, data)
    return {"message": "Feedback submitted successfully."}

@app.get("/feedback")
async def get_feedback(key: str = ""):
    if key and key != ADMIN_KEY:
        return {"error": "Unauthorized"}
    data = load_data(FEEDBACK_FILE)
    return {"feedback": data}
