# app.py
import os
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langdetect import detect
import difflib
from huggingface_hub import InferenceClient

from sqlalchemy import Column, Integer, String, Text, DateTime, select, func
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker

# ------------------ CONFIG ------------------
DEFAULT_DB = "postgresql+asyncpg://training_data_pqbq_user:9enoHVmpfOuYplmcRUL7ZN4Lygs5876D@dpg-d45g8om3jp1c73dguc4g-a/training_data_pqbq"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DB)
ADMIN_KEY = os.getenv("ADMIN_KEY", "mc250132689")
HF_TOKEN = os.getenv("HF_TOKEN")  # optional; set to enable HF generative responses

# HuggingFace Model Client (optional)
client = None
if HF_TOKEN:
    client = InferenceClient(provider="groq", api_key=HF_TOKEN)

# ------------------ DATABASE SETUP ------------------
Base = declarative_base()

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

engine = create_async_engine(DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# ------------------ FASTAPI APP ------------------
app = FastAPI(title="Identifying Spiritual Sickness Chatbot (Backend)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

class ChatRequest(BaseModel):
    message: str

class TrainRequest(BaseModel):
    question: str
    answer: str
    lang: Optional[str] = "en"

class FeedbackRequest(BaseModel):
    text: str

async def init_models():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

trained_answers = {}  # {lang: {question_lower: answer}}

async def load_memory_from_db(session: AsyncSession):
    trained_answers.clear()
    q = select(TrainingData)
    result = await session.execute(q)
    rows = result.scalars().all()
    for item in rows:
        lang = item.lang or "en"
        trained_answers.setdefault(lang, {})[item.question.lower()] = item.answer

@app.on_event("startup")
async def startup_event():
    await init_models()
    async with AsyncSessionLocal() as session:
        await load_memory_from_db(session)

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

@app.post("/chat")
async def chat(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    user_message = req.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Please type a message.")

    try:
        lang = detect(user_message)
    except:
        lang = "en"

    lower_msg = user_message.lower()
    symptoms = [label for key, label in SYMPTOM_KEYWORDS.items() if key in lower_msg]

    if symptoms:
        formatted = "\n- ".join(symptoms)
        reply = (
            "🕌 *Possible Spiritual Symptoms Noticed*\n\n"
            f"You mentioned signs related to:\n- {formatted}\n\n"
            "Recommended steps:\n"
            "1. Recite Surah Al-Baqarah daily\n"
            "2. Recite Ayat al-Kursi before sleep\n"
            "3. Play Ruqyah audio (Mishary Rashid / Saad Al-Ghamdi)\n"
            "4. Maintain wudu and reduce stress\n\n"
            "If symptoms persist or intensify, consult a *qualified ruqyah practitioner*."
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

    if client:
        try:
            completion = client.chat.completions.create(
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
        reply = (
            "Model not configured on server. Please set HF_TOKEN in environment to enable generative responses, "
            "or train this question via the admin Train page."
        )

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
    out = []
    for r in rows:
        out.append({
            "id": r.id,
            "question": r.question,
            "answer": r.answer,
            "lang": r.lang,
            "created_at": r.created_at.isoformat()
        })
    return {"training_data": out}

@app.get("/chat-logs")
async def get_chat_logs(key: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    if key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    q = select(ChatLog).order_by(ChatLog.created_at.desc()).limit(5000)
    res = await db.execute(q)
    rows = res.scalars().all()
    out = []
    for r in rows:
        out.append({
            "id": r.id,
            "user": r.user_message,
            "bot": r.bot_response,
            "lang": r.lang,
            "time": r.created_at.isoformat()
        })
    return {"logs": out}

@app.post("/feedback")
async def submit_feedback(payload: FeedbackRequest, db: AsyncSession = Depends(get_db)):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Feedback cannot be empty")
    fb = Feedback(text=payload.text.strip())
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
    out = [{"id": r.id, "text": r.text, "time": r.created_at.isoformat()} for r in rows]
    return {"feedback": out}

@app.get("/admin-stats")
async def admin_stats(key: str, db: AsyncSession = Depends(get_db)):
    if key != ADMIN_KEY:
        return {"error": "Unauthorized"}

    total_res = await db.execute(select(func.count(TrainingData.id)))
    total_records = total_res.scalar_one() or 0

    if total_records == 0:
        return {"total_records": 0, "avg_question_length": 0, "avg_answer_length": 0}

    avg_q_len_res = await db.execute(select(func.avg(func.length(TrainingData.question))))
    avg_q = int(avg_q_len_res.scalar_one() or 0)

    avg_a_len_res = await db.execute(select(func.avg(func.length(TrainingData.answer))))
    avg_a = int(avg_a_len_res.scalar_one() or 0)

    return {
        "total_records": int(total_records),
        "avg_question_length": avg_q,
        "avg_answer_length": avg_a,
    }

@app.delete("/delete-entry")
async def delete_entry(question: str, key: str, db: AsyncSession = Depends(get_db)):
    if key != ADMIN_KEY:
        return {"error": "Unauthorized"}

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
