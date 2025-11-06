import os
import difflib
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel
from langdetect import detect

from sqlalchemy import Column, Integer, String, Text, DateTime, select, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker

# ---------------- CONFIG ----------------
DEFAULT_SQLITE = "sqlite+aiosqlite:///./training.db"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_SQLITE).strip()
ADMIN_KEY = os.getenv("ADMIN_KEY", "mc250132689")

if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

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
templates = Jinja2Templates(directory="templates")

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
trained_answers = {}

# ---------------- Islamic rules & symptoms ----------------
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
    try:
        await session.execute(text("ALTER TABLE training_data ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"))
    except: pass
    try:
        await session.execute(text("ALTER TABLE chat_logs ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"))
    except: pass
    try:
        await session.execute(text("ALTER TABLE chat_logs ADD COLUMN IF NOT EXISTS lang VARCHAR(16)"))
    except: pass
    try:
        await session.execute(text("ALTER TABLE feedback ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"))
    except: pass

# ---------------- Startup ----------------
@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with AsyncSessionLocal() as s:
        await ensure_columns(s)
        await load_memory_from_db(s)

# ---------------- Endpoints ----------------
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/chat-page", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/train-page", response_class=HTMLResponse)
async def train_page(request: Request):
    async with AsyncSessionLocal() as db:
        q = select(TrainingData).order_by(TrainingData.created_at.desc())
        res = await db.execute(q)
        train_data = res.scalars().all()
    return templates.TemplateResponse("train.html", {"request": request, "train_data": train_data})

@app.get("/view_training-page", response_class=HTMLResponse)
async def view_training_page(request: Request):
    async with AsyncSessionLocal() as db:
        q = select(TrainingData).order_by(TrainingData.created_at.desc())
        res = await db.execute(q)
        train_data = res.scalars().all()
    return templates.TemplateResponse("view_training.html", {"request": request, "train_data": train_data})

@app.get("/chat_logs-page", response_class=HTMLResponse)
async def chat_logs_page(request: Request):
    async with AsyncSessionLocal() as db:
        q = select(ChatLog).order_by(ChatLog.created_at.desc())
        res = await db.execute(q)
        chat_logs = res.scalars().all()
    return templates.TemplateResponse("chat_logs.html", {"request": request, "chat_logs": chat_logs})

@app.get("/feedback-page", response_class=HTMLResponse)
async def feedback_page(request: Request):
    async with AsyncSessionLocal() as db:
        q = select(Feedback).order_by(Feedback.created_at.desc())
        res = await db.execute(q)
        feedback_data = res.scalars().all()
    return templates.TemplateResponse("feedback.html", {"request": request, "feedback_data": feedback_data})

@app.post("/chat")
async def chat(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    user_message = (req.message or "").strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Please type a message.")
    try:
        lang = detect(user_message)
    except:
        lang = "en"
    lower = user_message.lower()
    forbidden = ["crystal", "tarot", "reiki", "chakra", "zodiac", "astrology"]
    for t in forbidden:
        if t in lower:
            reply = ("I cannot advise on non-Islamic spiritual practices. "
                     "This assistant provides Islamic guidance based on Quran, Sunnah, ruqyah, and qualified practitioners.")
            db.add(ChatLog(user_message=user_message, bot_response=reply, lang=lang))
            await db.commit()
            return {"response": reply}

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

    lang_dict = trained_answers.get(lang, {})
    if lang_dict:
        close = difflib.get_close_matches(user_message.lower(), lang_dict.keys(), n=1, cutoff=0.6)
        if close:
            reply = lang_dict[close[0]]
            db.add(ChatLog(user_message=user_message, bot_response=reply, lang=lang))
            await db.commit()
            return {"response": reply}

    reply = ("Model not configured. Please train this question via the admin Train page.")
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

@app.post("/feedback")
async def submit_feedback(payload: FeedbackRequest, db: AsyncSession = Depends(get_db)):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Feedback cannot be empty.")
    fb = Feedback(text=payload.text.strip())
    db.add(fb)
    await db.commit()
    return {"message": "Thank you for your feedback!"}
