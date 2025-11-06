# app.py
import os
import difflib
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langdetect import detect

from sqlalchemy import Column, Integer, String, Text, DateTime, select, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker

# ---------------- Check for Jinja2 ----------------
try:
    from starlette.templating import Jinja2Templates
    templates = Jinja2Templates(directory="templates")
except ImportError:
    templates = None
    print("WARNING: jinja2 is not installed. HTML template rendering will not work. Run 'pip install jinja2'")

# ---------------- CONFIG ----------------
DEFAULT_SQLITE = "sqlite+aiosqlite:///./training.db"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_SQLITE).strip()
ADMIN_KEY = os.getenv("ADMIN_KEY", "mc250132689")
HF_TOKEN = os.getenv("HF_TOKEN")  # optional

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
ISLAMIC_RULES = """
You are an Islamic assistant specializing in:
- Spiritual sickness (jinn, sihr/black magic, evil eye)
- Ruqyah and Islamic medical practices
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
@app.get("/ping")
async def ping():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}
