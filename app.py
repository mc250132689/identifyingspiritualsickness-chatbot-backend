# app.py
import os
import difflib
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langdetect import detect
from huggingface_hub import InferenceClient

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, func, select
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker

# ---------------- CONFIG ----------------
DEFAULT_DB = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://training_data_pqbq_user:9enoHVmpfOuYplmcRUL7ZN4Lygs5876D@dpg-d45g8om3jp1c73dguc4g-a/training_data_pqbq"
)
DATABASE_URL = DEFAULT_DB  # must be postgresql+asyncpg://...
ADMIN_KEY = os.getenv("ADMIN_KEY", "mc250132689")
HF_TOKEN = os.getenv("HF_TOKEN")  # optional

# HuggingFace client (optional)
hf_client = None
if HF_TOKEN:
    hf_client = InferenceClient(provider="groq", api_key=HF_TOKEN)

# Use asyncpg driver and reasonable pool settings to avoid connections sleeping
# pool_pre_ping helps detect/refresh broken connections
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)

AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

# ---------------- MODELS ----------------
class TrainingData(Base):
    __tablename__ = "training_data"
    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False, index=True)
    answer = Column(Text, nullable=False)
    lang = Column(String(16), nullable=False, default="en")
    created_at = Column(DateTime, default=datetime.utcnow)

class ChatLog(Base):
    __tablename__ = "chat_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_message = Column(Text, nullable=False)
    bot_response = Column(Text, nullable=False)
    lang = Column(String(16), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

# ---------------- APP ----------------
app = FastAPI(title="Identifying Spiritual Sickness Chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initial in-memory cache for quick match lookups (language keyed)
trained_answers = {}

# Islamic rules (system prompt / enforcement)
ISLAMIC_RULES = """
You are an Islamic assistant specializing in: Islamic spiritual sickness (jinn disturbance, sihr/black magic),
Islamic medical practices & ruqyah from Qur'an & Sunnah.

RULES:
1) Follow Quran, authentic Sunnah, and mainstream scholars.
2) Do NOT suggest non-Islamic spiritual healing (crystals, tarot, reiki, yoga chakras, tarot).
3) Use conditional language when discussing possession or jinn.
4) Encourage ruqyah, dua, dhikr, salah, Quran recitation, and seeking qualified ruqyah practitioners.
"""

# Quick symptom keywords
SYMPTOM_KEYWORDS = {
    "nightmare": "Recurring bad dreams",
    "sleep paralysis": "Sleep paralysis episodes",
    "shadow": "Seeing black shadows",
    "jinn": "Possible jinn disturbance",
    "magic": "Possible sihr / black magic",
    "sihr": "Signs of sihr (black magic)",
    "ruqyah": "Seeking ruqyah guidance",
    "waswas": "Waswas (whispering doubts)"
}

# ---------------- Pydantic models ----------------
class ChatRequest(BaseModel):
    message: str

class TrainRequest(BaseModel):
    question: str
    answer: str
    lang: Optional[str] = "en"

class FeedbackRequest(BaseModel):
    text: str

# ---------------- UTILITIES ----------------
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

async def load_memory_from_db(session: AsyncSession):
    trained_answers.clear()
    q = select(TrainingData)
    res = await session.execute(q)
    rows = res.scalars().all()
    for r in rows:
        lang = r.lang or "en"
        trained_answers.setdefault(lang, {})[r.question.lower()] = r.answer

@app.on_event("startup")
async def startup():
    # Create tables if not exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    # load memory
    async with AsyncSessionLocal() as s:
        await load_memory_from_db(s)

# ---------------- HEALTH ----------------
@app.get("/ping")
async def ping():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

# ---------------- CHAT ----------------
@app.post("/chat")
async def chat(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    user_message = (req.message or "").strip()
    if not user_message:
        raise HTTPException(400, "Please type a message.")

    # detect language
    try:
        lang = detect(user_message)
    except:
        lang = "en"

    lower = user_message.lower()

    # Enforce topic restriction: if user asks non-Islamic spiritual modalities, refuse politely
    forbidden_terms = ["crystal", "crystals", "tarot", "reiki", "chakra", "chakra healing", "zodiac", "astrology", "witchcraft"]
    for t in forbidden_terms:
        if t in lower:
            reply = ("I cannot advise on non-Islamic spiritual practices (e.g., crystals, tarot, reiki, chakras, astrology). "
                     "This assistant focuses on Islamic guidance: Quran, Sunnah, ruqyah, dua, and consulting qualified practitioners.")
            await db.execute(
                select(ChatLog)  # no-op but keep consistent pattern
            )
            db.add(ChatLog(user_message=user_message, bot_response=reply, lang=lang))
            await db.commit()
            return {"response": reply}

    # Quick symptom detection
    symptoms = [label for k, label in SYMPTOM_KEYWORDS.items() if k in lower]
    if symptoms:
        formatted = "\n- ".join(symptoms)
        reply = (
            "🕌 *Possible Spiritual Symptoms Noticed*\n\n"
            f"You mentioned signs related to:\n- {formatted}\n\n"
            "Recommended steps:\n"
            "1. Recite Surah Al-Baqarah daily\n"
            "2. Recite Ayat al-Kursi before sleep\n"
            "3. Play authentic Ruqyah audio and maintain wudu\n"
            "4. Seek a qualified ruqyah practitioner if symptoms persist\n"
        )
        db.add(ChatLog(user_message=user_message, bot_response=reply, lang=lang))
        await db.commit()
        return {"response": reply}

    # Check learned answers (fuzzy)
    lang_dict = trained_answers.get(lang, {})
    if lang_dict:
        close = difflib.get_close_matches(user_message.lower(), lang_dict.keys(), n=1, cutoff=0.6)
        if close:
            reply = lang_dict[close[0]]
            db.add(ChatLog(user_message=user_message, bot_response=reply, lang=lang))
            await db.commit()
            return {"response": reply}

    # Use HF model if configured, else ask to train
    if hf_client:
        try:
            completion = hf_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": ISLAMIC_RULES},
                    {"role": "user", "content": user_message},
                ],
            )
            reply = completion.choices[0].message["content"]
        except Exception:
            reply = "⚠️ Error generating response. Please try again later."
    else:
        reply = ("Model not configured on server. You can train this question via admin -> Train, "
                 "or set HF_TOKEN to enable generative replies.")

    # Save as learned answer and chat log
    new = TrainingData(question=user_message, answer=reply, lang=lang)
    db.add(new)
    db.add(ChatLog(user_message=user_message, bot_response=reply, lang=lang))
    await db.commit()

    trained_answers.setdefault(lang, {})[user_message.lower()] = reply

    return {"response": reply}

# ---------------- TRAIN ----------------
@app.post("/train")
async def train(payload: TrainRequest, db: AsyncSession = Depends(get_db)):
    q = payload.question.strip()
    a = payload.answer.strip()
    lang = payload.lang or "en"
    if not q or not a:
        raise HTTPException(400, "Both question and answer are required.")
    item = TrainingData(question=q, answer=a, lang=lang)
    db.add(item)
    await db.commit()
    trained_answers.setdefault(lang, {})[q.lower()] = a
    return {"message": "Training data submitted", "id": item.id}

@app.get("/get-training-data")
async def get_training_data(db: AsyncSession = Depends(get_db)):
    q = select(TrainingData).order_by(TrainingData.created_at.desc())
    res = await db.execute(q)
    rows = res.scalars().all()
    out = [{"id": r.id, "question": r.question, "answer": r.answer, "lang": r.lang, "created_at": r.created_at.isoformat()} for r in rows]
    return {"training_data": out}

# ---------------- CHAT LOGS ----------------
@app.get("/chat-logs")
async def chat_logs(key: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    if key != ADMIN_KEY:
        raise HTTPException(401, "Unauthorized")
    q = select(ChatLog).order_by(ChatLog.created_at.desc()).limit(5000)
    res = await db.execute(q)
    rows = res.scalars().all()
    out = [{"id": r.id, "user": r.user_message, "bot": r.bot_response, "lang": r.lang, "time": r.created_at.isoformat()} for r in rows]
    return {"logs": out}

# ---------------- FEEDBACK ----------------
@app.post("/feedback")
async def submit_feedback(payload: FeedbackRequest, db: AsyncSession = Depends(get_db)):
    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(400, "Feedback cannot be empty")
    fb = Feedback(text=text)
    db.add(fb)
    await db.commit()
    return {"status": "success", "message": "Feedback submitted"}

@app.get("/feedback")
async def get_feedback(key: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    if key != ADMIN_KEY:
        raise HTTPException(401, "Unauthorized")
    q = select(Feedback).order_by(Feedback.created_at.desc()).limit(2000)
    res = await db.execute(q)
    rows = res.scalars().all()
    out = [{"id": r.id, "text": r.text, "time": r.created_at.isoformat()} for r in rows]
    return {"feedback": out}

# ---------------- ADMIN STATS / CRUD ----------------
@app.get("/admin-stats")
async def admin_stats(key: str, db: AsyncSession = Depends(get_db)):
    if key != ADMIN_KEY:
        return {"error": "Unauthorized"}
    total_res = await db.execute(select(func.count(TrainingData.id)))
    total = int(total_res.scalar_one() or 0)
    if total == 0:
        return {"total_records": 0, "avg_question_length": 0, "avg_answer_length": 0}
    avg_q_res = await db.execute(select(func.avg(func.length(TrainingData.question))))
    avg_a_res = await db.execute(select(func.avg(func.length(TrainingData.answer))))
    avg_q = int(avg_q_res.scalar_one() or 0)
    avg_a = int(avg_a_res.scalar_one() or 0)
    return {"total_records": total, "avg_question_length": avg_q, "avg_answer_length": avg_a}

@app.delete("/delete-entry")
async def delete_entry(question: str, key: str, db: AsyncSession = Depends(get_db)):
    if key != ADMIN_KEY:
        return {"error": "Unauthorized"}
    # case-insensitive find
    q = select(TrainingData).where(func.lower(TrainingData.question) == question.lower())
    res = await db.execute(q)
    item = res.scalars().first()
    if not item:
        return {"status": "failed", "message": "Entry not found"}
    await db.delete(item)
    await db.commit()
    await load_memory_from_db(db)
    return {"status": "success", "message": "Entry deleted"}

@app.put("/update-entry")
async def update_entry(question: str, new_question: str, new_answer: str, new_lang: str, key: str, db: AsyncSession = Depends(get_db)):
    if key != ADMIN_KEY:
        return {"error": "Unauthorized"}
    q = select(TrainingData).where(func.lower(TrainingData.question) == question.lower())
    res = await db.execute(q)
    item = res.scalars().first()
    if not item:
        return {"status": "failed", "message": "Entry not found"}
    item.question = new_question
    item.answer = new_answer
    item.lang = new_lang
    db.add(item)
    await db.commit()
    await load_memory_from_db(db)
    return {"status": "success", "message": "Entry updated"}
