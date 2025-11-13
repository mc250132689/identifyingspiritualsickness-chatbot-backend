# app.py
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
import tempfile
import shutil
from typing import List, Optional
import asyncio
import aiohttp
import subprocess
import datetime
import sqlite3


# ---------------------------
# Auto restore from GitHub
# ---------------------------
async def auto_restore_from_github():
    if not GH_TOKEN:
        print("[RESTORE] GH_TOKEN not set; skipping restore")
        return
    remote_url = f"https://x-access-token:{GH_TOKEN}@github.com/mc250132689/identifyingspiritualsickness-chatbot-backend.git"
    local_repo_dir = os.path.join(DATA_DIR, ".local_git")
    os.makedirs(local_repo_dir, exist_ok=True)
    if not os.path.exists(os.path.join(local_repo_dir, ".git")):
        await run_git_command(["git","init"], cwd=local_repo_dir)
        await run_git_command(["git","remote","add","origin",remote_url], cwd=local_repo_dir)
    # Fetch latest
    await run_git_command(["git","fetch","origin"], cwd=local_repo_dir)
    # Checkout branch
    await run_git_command(["git","checkout","-B", GITHUB_BRANCH, f"origin/{GITHUB_BRANCH}"], cwd=local_repo_dir)
    # Copy database.db if exists
    src = os.path.join(local_repo_dir, "database.db")
    dst = os.path.join(DATA_DIR, "database.db")
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print("[RESTORE] database.db restored from GitHub backup")
    else:
        print("[RESTORE] No database.db found in GitHub backup")

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
# Config
# ---------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(api_key=HF_TOKEN) if HF_TOKEN else None

GITHUB_REPO = "https://github.com/mc250132689/identifyingspiritualsickness-chatbot-backend.git"
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")
GH_TOKEN = os.getenv("GH_TOKEN")
APP_URL = os.getenv("APP_URL", "https://identifyingspiritualsickness-chatbot.onrender.com")
ADMIN_KEY = os.getenv("ADMIN_KEY", "mc250132689")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------
# SQLite setup
# ---------------------------
DB_FILE = os.path.join(DATA_DIR, "database.db")

def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    # Training data table
    c.execute('''
    CREATE TABLE IF NOT EXISTS training_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        lang TEXT NOT NULL,
        hadith_refs TEXT
    )
    ''')
    # Feedback table
    c.execute('''
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        data TEXT
    )
    ''')
    conn.commit()
    conn.close()

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
    c = conn.cursor()
    c.execute("SELECT * FROM training_data")
    rows = c.fetchall()
    global trained_answers
    trained_answers = {}
    for item in rows:
        lang = item["lang"]
        q = item["question"].strip()
        a = item["answer"]
        if lang not in trained_answers:
            trained_answers[lang] = {}
        trained_answers[lang][q.lower()] = {"answer": a, "norm": normalize_text(q)}
    conn.close()

load_data_into_memory()

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
    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT INTO training_data(question,answer,lang,hadith_refs) VALUES (?,?,?,?)",
              (user_message, reply, lang, json.dumps(hadith_refs)))
    conn.commit()
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
    c = conn.cursor()
    # Check existing
    c.execute("SELECT id FROM training_data WHERE lower(question)=? AND lang=?", (question.lower(), lang))
    row = c.fetchone()
    if row:
        c.execute("UPDATE training_data SET answer=?, hadith_refs=? WHERE id=?",
                  (answer, json.dumps(extract_hadith_refs(answer)), row["id"]))
        msg = "Updated training data successfully."
    else:
        c.execute("INSERT INTO training_data(question, answer, lang, hadith_refs) VALUES (?,?,?,?)",
                  (question, answer, lang, json.dumps(extract_hadith_refs(answer))))
        msg = "Added training data successfully."
    conn.commit()
    conn.close()
    # Update memory
    if lang not in trained_answers: trained_answers[lang] = {}
    trained_answers[lang][question.lower()] = {"answer": answer, "norm": normalize_text(question)}
    return {"message": msg}

@app.get("/get-training-data")
async def get_training_data():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM training_data")
    rows = c.fetchall()
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
    c = conn.cursor()
    c.execute("INSERT INTO feedback(data) VALUES (?)", (json.dumps(item.dict()),))
    conn.commit()
    conn.close()
    return {"message": "Feedback submitted. Jazakallah khair."}

@app.get("/export-feedback")
async def export_feedback(key: str = Query(None)):
    if key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM feedback")
    rows = c.fetchall()
    conn.close()
    out = [json.loads(r["data"]) for r in rows]
    return {"feedback": out}

# ---------------------------
# Health endpoint
# ---------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}

# ---------------------------
# Startup
# ---------------------------
@app.on_event("startup")
async def startup_event():
    load_data_into_memory()

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
    c = conn.cursor()
    c.execute("SELECT * FROM training_data")
    rows = c.fetchall()
    conn.close()
    keywords = HADITH_KEYWORDS
    results = []
    for item in rows:
        combined = f"{item['question']} {item['answer']}"
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
    c = conn.cursor()
    c.execute("SELECT * FROM training_data")
    data = c.fetchall()
    total = len(data)
    lang_count = Counter(item["lang"] for item in data)
    avg_q = round(sum(len(i["question"]) for i in data)/total, 1) if total else 0
    avg_a = round(sum(len(i["answer"]) for i in data)/total, 1) if total else 0
    hadith_count = sum(1 for i in data if i["hadith_refs"])
    hadith_examples = [dict(i) for i in data if i["hadith_refs"]][:5]
    q_counter = Counter(normalize_text(i["question"]) for i in data)
    top_questions = [q for q,_ in q_counter.most_common(10)]
    recent = [dict(i) for i in data[-10:]] if total else []

    # feedback count
    c.execute("SELECT * FROM feedback")
    fb = c.fetchall()
    feedback_count = len(fb)
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
# Background tasks: GitHub backup
# ---------------------------
async def run_git_command(cmd_args, cwd=None, check=False, env=None):
    def _run():
        try:
            res = subprocess.run(cmd_args, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check, text=True)
            return res.returncode, res.stdout, res.stderr
        except Exception as e:
            return 1, "", str(e)
    return await asyncio.to_thread(_run)

async def auto_backup_to_github():
    if not GH_TOKEN:
        print("[BACKUP] GH_TOKEN not set; skipping backup")
        return
    remote_url = f"https://x-access-token:{GH_TOKEN}@github.com/mc250132689/identifyingspiritualsickness-chatbot-backend.git"
    local_repo_dir = os.path.join(DATA_DIR, ".local_git")
    os.makedirs(local_repo_dir, exist_ok=True)
    if not os.path.exists(os.path.join(local_repo_dir, ".git")):
        await run_git_command(["git","init"], cwd=local_repo_dir)
        await run_git_command(["git","remote","add","origin",remote_url], cwd=local_repo_dir)
    while True:
        try:
            timestamp = datetime.datetime.utcnow().isoformat()
            for fname in ["database.db"]:
                src = os.path.join(DATA_DIR, fname)
                dst = os.path.join(local_repo_dir, fname)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
            await run_git_command(["git","add","."], cwd=local_repo_dir)
            await run_git_command(["git","commit","-m", f"Auto backup {timestamp}"], cwd=local_repo_dir, check=False)
            await run_git_command(["git","push","origin",f"HEAD:{GITHUB_BRANCH}"], cwd=local_repo_dir, check=False)
            print(f"[BACKUP] pushed backup at {timestamp}")
        except Exception as e:
            print("[BACKUP] exception:", e)
        await asyncio.sleep(1800)  # 30 minutes

# ---------------------------
# Startup hooks
# ---------------------------
@app.on_event("startup")
async def startup_event():
    if GH_TOKEN:
        await auto_restore_from_github()   # <-- auto-restore first
    load_data_into_memory()
    asyncio.create_task(keep_awake_task())
    if GH_TOKEN:
        asyncio.create_task(auto_backup_to_github())
    print("[STARTUP] Completed startup tasks")
