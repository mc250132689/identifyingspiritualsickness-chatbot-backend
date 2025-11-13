# app.py
from fastapi import FastAPI, Query, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from langdetect import detect
import difflib, json, os, requests, re, sqlite3, asyncio, aiohttp, shutil, subprocess, datetime
from typing import List, Optional

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True,
)

HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(api_key=HF_TOKEN) if HF_TOKEN else None

GITHUB_REPO = "https://github.com/mc250132689/identifyingspiritualsickness-chatbot-backend.git"
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")
GH_TOKEN = os.getenv("GH_TOKEN")
APP_URL = os.getenv("APP_URL", "https://identifyingspiritualsickness-chatbot.onrender.com")
ADMIN_KEY = os.getenv("ADMIN_KEY", "mc250132689")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
DB_FILE = os.path.join(DATA_DIR, "database.db")

# --------------------------------------------------
# DATABASE SETUP
# --------------------------------------------------
def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS training_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        lang TEXT NOT NULL,
        hadith_refs TEXT
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        data TEXT
    )''')
    conn.commit()
    conn.close()

init_db()

# --------------------------------------------------
# UTILITIES
# --------------------------------------------------
def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_hadith_refs(text: str) -> List[str]:
    refs = re.findall(r"(bukhari|muslim|tirmidhi|abu dawood|nasai|ibn majah|riyad)", text, flags=re.I)
    return list(set(refs))

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

# --------------------------------------------------
# MODELS
# --------------------------------------------------
class ChatRequest(BaseModel):
    message: str

class TrainRequest(BaseModel):
    question: str
    answer: str

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

# --------------------------------------------------
# ENDPOINTS
# --------------------------------------------------

@app.get("/")
async def root():
    return {"status": "✅ Spiritual Sickness Chatbot backend active."}


# -------------------------------
# 🧠 CHATBOT ENDPOINT
# -------------------------------
@app.post("/chat")
async def chat(req: ChatRequest):
    user_message = req.message.strip()
    if not user_message:
        return {"response": "Please type a message."}
    try:
        lang = detect(user_message)
    except:
        lang = "en"
    norm_user = normalize_text(user_message)
    lang_dict = trained_answers.get(lang, {})

    # Q&A lookup
    best_match, best_score = None, 0.0
    for q, meta in lang_dict.items():
        score = difflib.SequenceMatcher(None, norm_user, meta["norm"]).ratio()
        if score > best_score:
            best_score, best_match = score, q
    if best_score >= 0.70 and best_match:
        return {"response": lang_dict[best_match]["answer"]}

    # GPT fallback
    if client:
        try:
            completion = client.chat.completions.create(
                model="openai/gpt-oss-20b:groq",
                messages=[
                    {"role": "system", "content": "You are an Islamic spiritual sickness assistant based on Quran and Sahih Hadith."},
                    {"role": "user", "content": user_message}
                ]
            )
            reply = completion.choices[0].message["content"]
        except:
            reply = "Sorry, model backend currently unavailable. Please try later."
    else:
        reply = "Model client not configured."

    hadith_refs = extract_hadith_refs(reply)
    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT INTO training_data(question,answer,lang,hadith_refs) VALUES (?,?,?,?)",
              (user_message, reply, lang, json.dumps(hadith_refs)))
    conn.commit()
    conn.close()
    if lang not in trained_answers:
        trained_answers[lang] = {}
    trained_answers[lang][user_message.lower()] = {"answer": reply, "norm": norm_user}
    return {"response": reply, "hadith_refs": hadith_refs}


# -------------------------------
# 🏫 TRAINING ENDPOINT
# -------------------------------
@app.post("/train")
async def train(req: TrainRequest):
    q, a = req.question.strip(), req.answer.strip()
    if not q or not a:
        raise HTTPException(status_code=400, detail="Both question and answer required.")
    try:
        lang = detect(q)
    except:
        lang = "en"
    refs = extract_hadith_refs(a)
    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT INTO training_data(question,answer,lang,hadith_refs) VALUES (?,?,?,?)",
              (q, a, lang, json.dumps(refs)))
    conn.commit()
    conn.close()
    if lang not in trained_answers:
        trained_answers[lang] = {}
    trained_answers[lang][q.lower()] = {"answer": a, "norm": normalize_text(q)}
    return {"status": "success", "question": q, "answer": a}


# -------------------------------
# 🧭 GUIDANCE ENDPOINT
# -------------------------------
@app.post("/guidance")
async def guidance(symptoms: str = Form(...), language: str = Form("en")):
    suggestion = f"For {symptoms}, recite Surah Al-Baqarah and authentic ruqyah duas. Seek a trusted scholar."
    return {"guidance": suggestion}


# -------------------------------
# 💬 FEEDBACK ENDPOINT
# -------------------------------
@app.post("/submit-feedback")
async def submit_feedback(item: FeedbackItem):
    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT INTO feedback(data) VALUES (?)", (json.dumps(item.dict()),))
    conn.commit()
    conn.close()
    return {"status": "success", "message": "Feedback submitted successfully."}


@app.get("/view-feedback")
async def view_feedback():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM feedback")
    rows = [json.loads(r["data"]) for r in c.fetchall()]
    conn.close()
    return {"feedback": rows}


# -------------------------------
# 📊 ADMIN STATS ENDPOINT
# -------------------------------
@app.get("/admin-stats")
async def admin_stats(key: str = Query(...)):
    if key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM training_data")
    total_train = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM feedback")
    total_feedback = c.fetchone()[0]
    conn.close()
    return {"training_records": total_train, "feedback_records": total_feedback}


# -------------------------------
# 📖 HADITH SEARCH ENDPOINT
# -------------------------------
@app.get("/hadith-search")
async def hadith_search(query: str):
    results = []
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM training_data")
    rows = c.fetchall()
    for r in rows:
        if query.lower() in r["answer"].lower():
            results.append({
                "question": r["question"],
                "answer": r["answer"],
                "hadith_refs": json.loads(r["hadith_refs"] or "[]")
            })
    conn.close()
    return {"results": results}

# --------------------------------------------------
# AUTO RESTORE FROM GITHUB
# --------------------------------------------------
async def restore_from_github():
    if not GH_TOKEN:
        print("[RESTORE] GH_TOKEN not set; skipping restore.")
        return
    base = f"https://raw.githubusercontent.com/mc250132689/identifyingspiritualsickness-chatbot-backend/{GITHUB_BRANCH}/data"
    files = ["database.db", "training_data.json", "feedback.json"]
    for f in files:
        local = os.path.join(DATA_DIR, f)
        if os.path.exists(local):
            continue
        url = f"{base}/{f}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as r:
                    if r.status == 200:
                        content = await r.read()
                        with open(local, "wb") as fp:
                            fp.write(content)
                        print(f"[RESTORE] Restored {f} from GitHub.")
        except Exception as e:
            print(f"[RESTORE ERROR] {f}:", e)

# --------------------------------------------------
# KEEP AWAKE + AUTO BACKUP
# --------------------------------------------------
async def keep_awake():
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.get(APP_URL, timeout=10) as r:
                    print(f"[PING] {APP_URL} -> {r.status}")
            except Exception as e:
                print("[PING ERROR]", e)
            await asyncio.sleep(180)

async def run_git(cmd, cwd=None):
    def _r():
        res = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
        return res.returncode, res.stdout, res.stderr
    return await asyncio.to_thread(_r)

async def auto_backup():
    if not GH_TOKEN:
        print("[BACKUP] GH_TOKEN not set; skipping.")
        return
    remote = f"https://x-access-token:{GH_TOKEN}@github.com/mc250132689/identifyingspiritualsickness-chatbot-backend.git"
    repo_dir = os.path.join(DATA_DIR, ".repo")
    os.makedirs(repo_dir, exist_ok=True)
    if not os.path.exists(os.path.join(repo_dir, ".git")):
        await run_git(["git", "init"], cwd=repo_dir)
        await run_git(["git", "remote", "add", "origin", remote], cwd=repo_dir)
    while True:
        try:
            ts = datetime.datetime.utcnow().isoformat()
            for f in ["database.db", "training_data.json", "feedback.json"]:
                src = os.path.join(DATA_DIR, f)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(repo_dir, f))
            await run_git(["git", "add", "."], cwd=repo_dir)
            await run_git(["git", "commit", "-m", f"Auto backup {ts}"], cwd=repo_dir)
            await run_git(["git", "push", "origin", f"HEAD:{GITHUB_BRANCH}"], cwd=repo_dir)
            print(f"[BACKUP] Auto backup completed {ts}")
        except Exception as e:
            print("[BACKUP ERROR]", e)
        await asyncio.sleep(1800)

# --------------------------------------------------
# STARTUP EVENTS
# --------------------------------------------------
@app.on_event("startup")
async def startup_event():
    await restore_from_github()
    load_data_into_memory()
    asyncio.create_task(keep_awake())
    asyncio.create_task(auto_backup())
    print("[STARTUP] ✅ Chatbot backend fully started.")

