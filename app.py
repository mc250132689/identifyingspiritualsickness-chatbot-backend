# app.py (Postgres / Neon ready)
from fastapi import FastAPI, Query, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from langdetect import detect
import difflib
import json
import os
import requests
from collections import Counter
import re
from typing import List, Optional
import asyncio
import aiohttp
import datetime
import psycopg2
import psycopg2.extras
import sqlite3  # only used by the local migration helper

# ---------------------------
# Basic app + CORS
# ---------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ---------------------------
# Config / HF client
# ---------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(api_key=HF_TOKEN) if HF_TOKEN else None

APP_URL = os.getenv("APP_URL", "https://identifyingspiritualsickness-chatbot.onrender.com")
ADMIN_KEY = os.getenv("ADMIN_KEY", "mc250132689")

# Postgres connection config (prefer DATABASE_URL)
DATABASE_URL = os.getenv("DATABASE_URL")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_PORT = os.getenv("DB_PORT", "5432")
CHANNEL_BINDING = os.getenv("CHANNEL_BINDING")  # e.g. "require"
SSLMODE = os.getenv("SSLMODE", "require")

# ---------------------------
# PostgreSQL helpers
# ---------------------------
def build_dsn_from_env():
    """Return a DSN string for psycopg2 from env vars, or DATABASE_URL if provided."""
    if DATABASE_URL:
        # psycopg2 accepts the URL directly
        return DATABASE_URL
    # Build a DSN; include sslmode and channel_binding if provided
    params = {
        "user": DB_USER,
        "password": DB_PASS,
        "host": DB_HOST,
        "port": DB_PORT,
        "dbname": DB_NAME,
    }
    dsn_parts = []
    for k, v in params.items():
        if v is not None and v != "":
            dsn_parts.append(f"{k}={v}")
    if SSLMODE:
        dsn_parts.append(f"sslmode={SSLMODE}")
    if CHANNEL_BINDING:
        # psycopg2 supports channel_binding as a libpq param
        dsn_parts.append(f"channel_binding={CHANNEL_BINDING}")
    return " ".join(dsn_parts)

def get_db():
    """
    Returns a new psycopg2 connection. Caller must close().
    Uses DATABASE_URL if provided, otherwise builds DSN from DB_* env vars.
    """
    dsn = build_dsn_from_env()
    # psycopg2.connect accepts both a libpq-style DSN string and a URL
    conn = psycopg2.connect(dsn)
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS training_data (
        id SERIAL PRIMARY KEY,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        lang TEXT NOT NULL,
        hadith_refs TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id SERIAL PRIMARY KEY,
        data JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    cur.close()
    conn.close()

# Attempt to create tables at start (idempotent)
init_db()

# ---------------------------
# Utilities
# ---------------------------
_non_alnum_re = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_multi_space_re = re.compile(r"\s+", flags=re.UNICODE)
def normalize_text(s: str) -> str:
    s = s or ""
    s = s.lower()
    s = _non_alnum_re.sub(" ", s)
    s = _multi_space_re.sub(" ", s).strip()
    return s

HADITH_REF_RE = re.compile(r"(bukhari|sahih al-bukhari|sahih muslim|muslim|tirmidhi|abu dawood|nasai|ibn majah|riyad)\b", flags=re.IGNORECASE)
HADITH_KEYWORDS = ["bukhari","muslim","sahih","riyad","tirmidhi","abu dawood","nasai","ibn majah","hadith","riyadh","sahih al-bukhari","sahih muslim","malik"]

def extract_hadith_refs(text: str) -> List[str]:
    refs = set()
    if not text: return []
    for m in HADITH_REF_RE.finditer(text):
        refs.add(m.group(0).strip())
    lowered = text.lower()
    for kw in HADITH_KEYWORDS:
        if kw in lowered: refs.add(kw)
    return list(refs)

async def hf_symptom_classify(text: str, model_id: str):
    try:
        if client:
            result = await asyncio.to_thread(lambda: client.text_classification(model=model_id, inputs=text))
            if result:
                return result[0]["label"].lower(), result[0]["score"]
    except Exception:
        pass
    return "none", 0.0

# ---------------------------
# Pydantic models
# ---------------------------
class ChatRequest(BaseModel):
    message: str

class TrainRequest(BaseModel):
    question: str
    answer: str

class GuidanceRequest(BaseModel):
    symptoms: Optional[str] = None
    details: Optional[str] = None
    language: Optional[str] = "en"

class FeedbackItem(BaseModel):
    name: Optional[str] = None
    student_id: Optional[str] = None
    program: Optional[str] = None
    q4: Optional[str] = None
    q5: Optional[str] = None
    q6: Optional[str] = None
    q7: Optional[str] = None
    q8: Optional[str] = None
    q9: Optional[str] = None
    q10: Optional[str] = None
    q11: Optional[str] = None
    q12: Optional[str] = None
    q13: Optional[str] = None
    q14: Optional[str] = None
    comments: Optional[str] = None

# ---------------------------
# In-memory cache for training
# ---------------------------
trained_answers = {}

def load_data_into_memory():
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT * FROM training_data ORDER BY id ASC")
    rows = cur.fetchall()
    global trained_answers
    trained_answers = {}
    for item in rows:
        lang = item["lang"]
        q = (item["question"] or "").strip()
        a = item["answer"]
        if lang not in trained_answers:
            trained_answers[lang] = {}
        trained_answers[lang][q.lower()] = {"answer": a, "norm": normalize_text(q)}
    cur.close()
    conn.close()

# load to memory on import (safe)
try:
    load_data_into_memory()
except Exception as e:
    print("[STARTUP] load_data_into_memory error:", e)

# ---------------------------
# Chat endpoint
# ---------------------------
@app.post("/chat")
async def chat(req: ChatRequest):
    user_message = req.message.strip()
    if not user_message: return {"response": "Please type a message."}
    try:
        lang = detect(user_message)
    except:
        lang = "en"
    norm_user = normalize_text(user_message)
    lang_dict = trained_answers.get(lang, {})

    # Step 1: trained answers matching
    best_match, best_score = None, 0.0
    for original_q, meta in lang_dict.items():
        score = difflib.SequenceMatcher(None, norm_user, meta["norm"]).ratio()
        if score > best_score:
            best_score = score
            best_match = original_q
    if best_score >= 0.70 and best_match:
        return {"response": lang_dict[best_match]["answer"]}

    # Step 2: symptom detection
    symptom_keywords = ["voices","see","visions","insomnia","nightmares","dizziness","palpitation","itching","sudden pain","fear","not myself"]
    is_symptoms = any(k in norm_user for k in symptom_keywords)
    label, score = "none", 0.0
    MODEL_ID = "your-hf-symptoms-classifier"

    if is_symptoms:
        text_for_model = user_message
        label, score = await hf_symptom_classify(text_for_model, MODEL_ID)

    if score >= 0.6:
        guidance_labels = {"jin":"jin involvement","sihir":"possible sihr","ruqyah":"symptoms treated with ruqyah"}
        suggestions = [f"Detected symptoms suggest {guidance_labels.get(label,'general ruqyah guidance')}. Consider seeking a qualified ruqyah practitioner and medical check-up."]
        return {"response":"⚠️ Symptom-based guidance provided.","model_label":label,"model_confidence":score,"suggestions":suggestions}

    # Step 3: GPT-OSS fallback
    if client:
        try:
            completion = client.chat.completions.create(
                model="openai/gpt-oss-20b:groq",
                messages=[
                    {"role":"system","content":"You are an Islamic knowledge assistant specializing in spiritual sickness, jin possession, sihr, ruqyah and Islamic dream interpretation. Answers MUST reference Quran, Sahih Hadith, valid ruqyah practices only."},
                    {"role":"user","content":user_message}
                ]
            )
            reply = completion.choices[0].message["content"]
        except:
            reply = "Sorry, model backend currently unavailable. Please try later."
    else:
        reply = "Model client not configured on server."

    hadith_refs = extract_hadith_refs(reply)
    # persist to postgres
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO training_data (question, answer, lang, hadith_refs) VALUES (%s, %s, %s, %s)",
        (user_message, reply, lang, json.dumps(hadith_refs))
    )
    conn.commit()
    cur.close()
    conn.close()

    # update memory cache
    if lang not in trained_answers: trained_answers[lang] = {}
    trained_answers[lang][user_message.lower()] = {"answer": reply, "norm": normalize_text(user_message)}

    return {"response": reply, "hadith_refs": hadith_refs}

# ---------------------------
# Training endpoints
# ---------------------------
@app.post("/train")
async def train(req: TrainRequest):
    question, answer = req.question.strip(), req.answer.strip()
    if not question or not answer:
        return {"message": "Please provide both question and answer."}
    try:
        lang = detect(question)
    except:
        lang = "en"

    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT id FROM training_data WHERE lower(question)=%s AND lang=%s", (question.lower(), lang))
    row = cur.fetchone()
    if row:
        cur.execute("UPDATE training_data SET answer=%s, hadith_refs=%s WHERE id=%s",
                    (answer, json.dumps(extract_hadith_refs(answer)), row["id"]))
        msg = "Updated training data successfully."
    else:
        cur.execute("INSERT INTO training_data (question, answer, lang, hadith_refs) VALUES (%s, %s, %s, %s)",
                    (question, answer, lang, json.dumps(extract_hadith_refs(answer))))
        msg = "Added training data successfully."
    conn.commit()
    cur.close()
    conn.close()

    # Update memory
    if lang not in trained_answers: trained_answers[lang] = {}
    trained_answers[lang][question.lower()] = {"answer": answer, "norm": normalize_text(question)}
    return {"message": msg}

@app.get("/get-training-data")
async def get_training_data():
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT * FROM training_data ORDER BY id ASC")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    data = []
    for r in rows:
        data.append({
            "id": r["id"],
            "question": r["question"],
            "answer": r["answer"],
            "lang": r["lang"],
            "hadith_refs": json.loads(r["hadith_refs"]) if r["hadith_refs"] else []
        })
    return {"training_data": data}

@app.post("/submit-feedback")
async def submit_feedback(item: FeedbackItem):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("INSERT INTO feedback (data) VALUES (%s)", (json.dumps(item.dict()),))
    conn.commit()
    cur.close()
    conn.close()
    return {"message": "Feedback submitted. Jazakallah khair."}

@app.get("/export-feedback")
async def export_feedback(key: str = Query(None)):
    if key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT * FROM feedback ORDER BY id ASC")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    out = [r["data"] for r in rows]
    return {"feedback": out}

# ---------------------------
# Guidance endpoint
# ---------------------------
@app.post("/guidance")
async def guidance(req: GuidanceRequest):
    symptoms = (req.symptoms or "") + " " + (req.details or "")
    s = normalize_text(symptoms)
    label, score = "none", 0.0
    lang = "en"
    if symptoms:
        try:
            lang = detect(symptoms)
        except:
            lang = "en"
        text_for_model = symptoms
        MODEL_ID = "your-hf-symptoms-classifier"
        label, score = await hf_symptom_classify(text_for_model, MODEL_ID)

    threshold = 0.6
    if score < threshold or not symptoms:
        keywords_jin = ["voices","hear voices","see","seeing","visions","speaking","possession","control me","not myself","sudden change"]
        keywords_sihir = ["sudden illness","sudden poverty","bad luck","marriage problem","family problem","sudden hatred","sudden fear"]
        keywords_ruqyah = ["insomnia","nightmares","sleepless","dizziness","palpitation","weird smell","itching","sudden pain"]
        matched_jin = any(k in s for k in keywords_jin)
        matched_sihir = any(k in s for k in keywords_sihir)
        matched_ruqyah = any(k in s for k in keywords_ruqyah)
    else:
        matched_jin = label == "jin"
        matched_sihir = label == "sihir"
        matched_ruqyah = label == "ruqyah"

    suggestions = []
    severity = "low"
    if matched_jin:
        severity = "high"
        suggestions.append("Signs suggest possible jin involvement. Seek qualified ruqyah practitioner & medical opinion.")
    if matched_sihir:
        severity = "medium" if severity != "high" else severity
        suggestions.append("Signs suggest possible sihr. Document, seek ruqyah & verify with scholars.")
    if matched_ruqyah:
        severity = "medium"
        suggestions.append("Symptoms match ruqyah-treated cases. Follow Quranic ruqyah steps.")
    if not (matched_jin or matched_sihir or matched_ruqyah):
        suggestions.append("Symptoms not strongly suggestive of jin/sihr. Seek medical check, consult scholar.")
        severity = "low"

    steps = [
        "1) Seek immediate medical check-up.",
        "2) Consult qualified ruqyah practitioner using Quran & authentic hadith.",
        "3) Maintain adhkar, prayer, Surah Al-Fatihah, Ayat al-Kursi, last two surahs.",
        "4) Avoid unqualified practitioners or un-Islamic practices."
    ]

    return {
        "severity": severity,
        "matched_jin": matched_jin,
        "matched_sihir": matched_sihir,
        "matched_ruqyah": matched_ruqyah,
        "suggestions": suggestions,
        "recommended_steps": steps,
        "model_label": label,
        "model_confidence": score,
        "language_detected": lang
    }

# ---------------------------
# Hadith search endpoint
# ---------------------------
@app.get("/hadith-search")
async def hadith_search(q: str = Query(...)):
    qnorm = normalize_text(q)
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT * FROM training_data ORDER BY id ASC")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    keywords = HADITH_KEYWORDS
    results = []
    for item in rows:
        combined = f"{item['question'] or ''} {item['answer'] or ''}"
        if qnorm in normalize_text(combined) or any(kw in combined.lower() for kw in keywords):
            results.append({
                "id": item["id"],
                "question": item["question"],
                "answer": item["answer"],
                "lang": item["lang"],
                "hadith_refs": json.loads(item["hadith_refs"]) if item["hadith_refs"] else []
            })
    return {"query": q, "count": len(results), "results": results}

# ---------------------------
# Admin stats endpoint
# ---------------------------
@app.get("/admin-stats")
async def admin_stats(key: str = Query(...)):
    if key != ADMIN_KEY:
        return {"error": "Unauthorized"}
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT * FROM training_data ORDER BY id ASC")
    data = cur.fetchall()
    total = len(data)
    lang_count = Counter(item["lang"] for item in data)
    avg_q = round(sum(len(i["question"]) for i in data)/total, 1) if total else 0
    avg_a = round(sum(len(i["answer"]) for i in data)/total, 1) if total else 0
    hadith_count = sum(1 for i in data if i["hadith_refs"])
    hadith_examples = [dict(i) for i in data if i["hadith_refs"]][:5]
    q_counter = Counter(normalize_text(i["question"]) for i in data)
    top_questions = [q for q,_ in q_counter.most_common(10)]
    recent = [dict(i) for i in data[-10:]] if total else []

    cur.execute("SELECT * FROM feedback ORDER BY id ASC")
    fb = cur.fetchall()
    feedback_count = len(fb)
    cur.close()
    conn.close()

    return {
        "total_records": total,
        "languages": dict(lang_count),
        "avg_question_length": avg_q,
        "avg_answer_length": avg_a,
        "hadith_count": hadith_count,
        "hadith_examples": hadith_examples,
        "top_questions": top_questions,
        "recent_records": recent,
        "feedback_count": feedback_count
    }

# ---------------------------
# Health / Ping endpoints
# ---------------------------
@app.get("/ping")
async def ping():
    return {"status": "ok", "message": "Server is awake"}

# ---------------------------
# Background tasks: keep-alive ping
# ---------------------------
async def keep_awake_task():
    session = aiohttp.ClientSession()
    try:
        while True:
            try:
                async with session.get(APP_URL, timeout=15) as r:
                    print(f"[KEEP_ALIVE] ping {APP_URL} -> {r.status}")
            except Exception as e:
                print("[KEEP_ALIVE] error pinging APP_URL:", e)
            await asyncio.sleep(180)  # 3 minutes
    finally:
        await session.close()

# ---------------------------
# Local migration helper (run manually, only if you have a local copy of the sqlite DB)
# ---------------------------
def migrate_sqlite_to_postgres(sqlite_path: str):
    """
    One-shot helper to migrate your local SQLite database (data/database.db)
    into the Postgres DB configured by env vars. Run this locally where your
    sqlite file exists, with DATABASE_URL or DB_* env vars set.
    """
    if not os.path.exists(sqlite_path):
        print("SQLite file not found:", sqlite_path)
        return
    # read sqlite data
    sconn = sqlite3.connect(sqlite_path)
    scur = sconn.cursor()
    scur.execute("SELECT id, question, answer, lang, hadith_refs FROM training_data")
    rows = scur.fetchall()
    scur.close()
    sconn.close()

    # push to postgres
    pconn = get_db()
    pcur = pconn.cursor()
    count = 0
    for r in rows:
        q = r[1] or ""
        a = r[2] or ""
        lang = r[3] or "en"
        hadith_refs = r[4] or None
        pcur.execute(
            "INSERT INTO training_data (question, answer, lang, hadith_refs) VALUES (%s, %s, %s, %s)",
            (q, a, lang, hadith_refs)
        )
        count += 1
    pconn.commit()
    pcur.close()
    pconn.close()
    print(f"Migrated {count} rows from sqlite -> postgres")

# ---------------------------
# Startup hooks
# ---------------------------
@app.on_event("startup")
async def startup_event():
    # reload memory into in-memory cache
    try:
        load_data_into_memory()
    except Exception as e:
        print("[STARTUP] load_data_into_memory error:", e)

    # optional keep-awake
    asyncio.create_task(keep_awake_task())
    print("[STARTUP] Completed startup tasks (Postgres mode)")

# If you want to run the migration locally:
# Set DATABASE_URL (or DB_* env vars) locally then run:
# python app.py --migrate-sqlite /path/to/data/database.db
if __name__ == "__main__":
    import sys
    if "--migrate-sqlite" in sys.argv:
        try:
            idx = sys.argv.index("--migrate-sqlite")
            sqlite_path = sys.argv[idx+1]
            migrate_sqlite_to_postgres(sqlite_path)
        except Exception as e:
            print("Usage: python app.py --migrate-sqlite /path/to/database.db")
            print("Error:", e)
