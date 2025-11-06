# app.py
import os
import difflib
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langdetect import detect

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, func, select, text
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker

# ---------------- CONFIG ----------------
# Default: use local SQLite for easy run-from-github
DEFAULT_SQLITE = "sqlite+aiosqlite:///./training.db"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_SQLITE).strip()
ADMIN_KEY = os.getenv("ADMIN_KEY", "mc250132689")
HF_TOKEN = os.getenv("HF_TOKEN")  # optional; not required

# If user provided a postgres-like URL without +asyncpg, try to auto-fix
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

# Create async engine
engine = create_async_engine(DATABASE_URL, echo=False, future=True)
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

# ---------------- Pydantic ----------------
class ChatRequest(BaseModel):
    message: str

class TrainRequest(BaseModel):
    question: str
    answer: str
    lang: Optional[str] = "en"

class FeedbackRequest(BaseModel):
    text: str

# ---------------- In-memory cache ----------------
trained_answers = {}  # {lang: {question_lower: answer}}

# ---------------- Islamic rules ----------------
ISLAMIC_RULES = """
You are an Islamic assistant specializing in:
- Spiritual sickness (jinn disturbance, sihr/black magic, evil eye)
- Islamic medical practices & ruqyah from Qur'an & Sunnah

RULES:
1. Follow Quran and authentic Sunnah.
2. Do NOT recommend non-Islamic spiritual methods (crystals, tarot, reiki, chakras, astrology).
3. Use conditional language for suspected possession; recommend qualified ruqyah practitioners when needed.
"""

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

# ---------------- Utilities ----------------
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

async def ensure_columns(session: AsyncSession):
    """
    Try to add created_at / lang columns if missing.
    This uses raw SQL and ignores errors (safe for existing columns).
    """
    # Add created_at to training_data
    try:
        # For Postgres: ALTER TABLE ... ADD COLUMN IF NOT EXISTS ...
        await session.execute(text("ALTER TABLE training_data ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"))
    except Exception:
        # SQLite may not support IF NOT EXISTS for columns; try simpler ALTER and ignore failures
        try:
            await session.execute(text("ALTER TABLE training_data ADD COLUMN created_at TIMESTAMP"))
        except Exception:
            pass

    # chat_logs created_at and lang
    try:
        await session.execute(text("ALTER TABLE chat_logs ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"))
    except Exception:
        try:
            await session.execute(text("ALTER TABLE chat_logs ADD COLUMN created_at TIMESTAMP"))
        except Exception:
            pass
    try:
        await session.execute(text("ALTER TABLE chat_logs ADD COLUMN IF NOT EXISTS lang VARCHAR(16)"))
    except Exception:
        try:
            await session.execute(text("ALTER TABLE chat_logs ADD COLUMN lang VARCHAR(16)"))
        except Exception:
            pass

    # feedback created_at
    try:
        await session.execute(text("ALTER TABLE feedback ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"))
    except Exception:
        try:
            await session.execute(text("ALTER TABLE feedback ADD COLUMN created_at TIMESTAMP"))
        except Exception:
            pass

# ---------------- Startup ----------------
@app.on_event("startup")
async def startup():
    # Create tables if not exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Try to add missing columns (safe)
    async with AsyncSessionLocal() as s:
        await ensure_columns(s)
        await load_memory_from_db(s)

# ---------------- Endpoints ----------------
@app.get("/ping")
async def ping():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

@app.post("/chat")
async def chat(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    user_message = (req.message or "").strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Please type a message.")

    try:
        lang = detect(user_message)
    except Exception:
        lang = "en"

    lower = user_message.lower()

    # Block non-Islamic spiritual modalities
    forbidden = ["crystal", "crystals", "tarot", "reiki", "chakra", "chakras", "zodiac", "astrology"]
    for t in forbidden:
        if t in lower:
            reply = ("I cannot advise on non-Islamic spiritual practices such as crystals, tarot, reiki, chakras or astrology. "
                     "This assistant provides Islamic guidance based on Quran, Sunnah, ruqyah, and qualified practitioners.")
            db.add(ChatLog(user_message=user_message, bot_response=reply, lang=lang))
            await db.commit()
            return {"response": reply}

    # Symptom quick-check
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

    # Check trained answers (fuzzy)
    lang_dict = trained_answers.get(lang, {})
    if lang_dict:
        close = difflib.get_close_matches(user_message.lower(), lang_dict.keys(), n=1, cutoff=0.6)
        if close:
            reply = lang_dict[close[0]]
            db.add(ChatLog(user_message=user_message, bot_response=reply, lang=lang))
            await db.commit()
            return {"response": reply}

    # Generative fall-back (disabled if no HF_TOKEN)
    reply = ("Model not configured on server. Please train this question via the admin Train page, "
             "or set HF_TOKEN in environment to enable generative responses.")
    # If you set HF integration, you can insert a call here to the HF API.

    # Save learned answer
    new_item = TrainingData(question=user_message, answer=reply, lang=lang)
    db.add(new_item)
    db.add(ChatLog(user_message=user_message, bot_response=reply, lang=lang))
    await db.commit()
    trained_answers.setdefault(lang, {})[user_message.lower()] = reply

    return {"response": reply}

@app.post("/train")
async def train_item(payload: TrainRequest, db: AsyncSession = Depends(get_db)):
    if not payload.question.strip() or not payload.answer.strip():
        raise HTTPException(status_code=400, detail="Both question and answer are required.")
    lang = payload.lang or "en"
    item = TrainingData(question=payload.question.strip(), answer=payload.answer.strip(), lang=lang)
    db.add(item)
    await db.commit()
    trained_answers.setdefault(lang, {})[payload.question.strip().lower()] = payload.answer.strip()
    return {"message": "Training data submitted!", "id": item.id}

@app.get("/get-training-data")
async def get_training_data(db: AsyncSession = Depends(get_db)):
    q = select(TrainingData).order_by(TrainingData.created_at.desc())
    res = await db.execute(q)
    rows = res.scalars().all()
    out = [{"id": r.id, "question": r.question, "answer": r.answer, "lang": r.lang, "created_at": r.created_at.isoformat() if r.created_at else None} for r in rows]
    return {"training_data": out}

@app.get("/chat-logs")
async def get_chat_logs(key: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    if key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    q = select(ChatLog).order_by(ChatLog.created_at.desc()).limit(5000)
    res = await db.execute(q)
    rows = res.scalars().all()
    out = [{"id": r.id, "user": r.user_message, "bot": r.bot_response, "lang": r.lang, "time": r.created_at.isoformat() if r.created_at else None} for r in rows]
    return {"logs": out}

@app.post("/feedback")
async def submit_feedback(payload: FeedbackRequest, db: AsyncSession = Depends(get_db)):
    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Feedback cannot be empty")
    fb = Feedback(text=text)
    db.add(fb)
    await db.commit()
    return {"status": "success", "message": "Feedback submitted"}

@app.get("/feedback")
async def get_feedback(key: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    if key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    q = select(Feedback).order_by(Feedback.created_at.desc()).limit(2000)
    res = await db.execute(q)
    rows = res.scalars().all()
    out = [{"id": r.id, "text": r.text, "time": r.created_at.isoformat() if r.created_at else None} for r in rows]
    return {"feedback": out}

@app.delete("/delete-entry")
async def delete_entry(question: str, key: str, db: AsyncSession = Depends(get_db)):
    if key != ADMIN_KEY:
        return {"error": "Unauthorized"}
    # case-insensitive match
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
