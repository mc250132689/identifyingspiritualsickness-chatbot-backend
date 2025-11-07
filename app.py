import os
import json
import asyncio
import re
from datetime import datetime, timedelta
from collections import Counter

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from langdetect import detect
from passlib.hash import bcrypt
from jose import JWTError, jwt
from huggingface_hub import InferenceClient
import requests
import difflib

# --- Config ---
DATA_DIR = "."
TRAIN_FILE = os.path.join(DATA_DIR, "training_data.json")
CHAT_FILE = os.path.join(DATA_DIR, "chat_history.json")
USERS_FILE = os.path.join(DATA_DIR, "users.json")

HF_TOKEN = os.getenv("HF_TOKEN", "")
ADMIN_JWT_SECRET = os.getenv("ADMIN_JWT_SECRET", "super-secret-change-me")
ADMIN_JWT_ALG = "HS256"
ADMIN_JWT_EXP_MINUTES = 1440  # 24 hours by default
DEFAULT_ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
DEFAULT_ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", None)  # if set, create admin on startup

# Allowed topic keywords (only respond to these general topics)
ALLOWED_TOPICS = [
    "jinn", "jin", "sihr", "sihir", "sihir", "sihir",
    "sihir", "magic", "black magic", "sihir", "sihr",
    "ruqyah", "ruqya", "ruqyah syar'iyyah", "spiritual",
    "spiritual sickness", "waswas", "evil eye", "ain", "hasad",
    "dream", "dreams", "interpretation", "interpretation of dreams",
    "mas", "mass", "possession"
]

# Simple regex for disallowed topics detection (e.g., politics, illegal, medical)
DISALLOWED_PATTERNS = [
    r"\b(bank|account|credit|card|transfer)\b",
    r"\b(how to make|build bomb|explosive|weapon|attack)\b",
    r"\b(illegal|hack|crack|pirate|piracy)\b",
    r"\b(diagnosis|prescribe|prescription|medical)\b"  # medical advice -> refer to doctor
]

# --- FastAPI setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- HF Inference client (if HF token provided) ---
client = None
if HF_TOKEN:
    client = InferenceClient(api_key=HF_TOKEN)

# --- Ensure data files exist ---
def ensure_file(path, default):
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default, f, ensure_ascii=False, indent=2)

ensure_file(TRAIN_FILE, {"training_data": []})
ensure_file(CHAT_FILE, {"chats": []})
ensure_file(USERS_FILE, {"users": []})

# --- Utility load/save ---
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# --- In-memory caches ---
trained_answers = {}  # trained_answers[lang][question_lower] = answer
def load_training_to_memory():
    data = load_json(TRAIN_FILE)["training_data"]
    global trained_answers
    trained_answers = {}
    for item in data:
        lang = item.get("lang", "en")
        q = item["question"].lower()
        a = item["answer"]
        trained_answers.setdefault(lang, {})[q] = a
    return data

load_training_to_memory()

# --- User management (JSON, bcrypt) ---
def load_users():
    return load_json(USERS_FILE)["users"]

def save_user(username, hashed):
    data = load_json(USERS_FILE)
    users = data.get("users", [])
    for u in users:
        if u["username"] == username:
            u["password"] = hashed
            save_json(USERS_FILE, data)
            return
    users.append({"username": username, "password": hashed})
    save_json(USERS_FILE, data)

# If default admin password is provided via env, create admin on startup
if DEFAULT_ADMIN_PASSWORD:
    hashed = bcrypt.hash(DEFAULT_ADMIN_PASSWORD)
    save_user(DEFAULT_ADMIN_USERNAME, hashed)

# --- Authentication helpers ---
def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ADMIN_JWT_EXP_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, ADMIN_JWT_SECRET, algorithm=ADMIN_JWT_ALG)
    return encoded_jwt

def verify_token(token: str):
    try:
        payload = jwt.decode(token, ADMIN_JWT_SECRET, algorithms=[ADMIN_JWT_ALG])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

async def admin_required(token: str = Query(..., description="Admin JWT token")):
    return verify_token(token)

# --- SSE: live streams for admin dashboard ---
app.state.sse_queues = []

async def sse_generator(q: asyncio.Queue):
    try:
        while True:
            data = await q.get()
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    except asyncio.CancelledError:
        return

@app.get("/stream")
async def stream_events():
    q = asyncio.Queue()
    app.state.sse_queues.append(q)
    async def gen():
        try:
            async for chunk in sse_generator(q):
                yield chunk
        finally:
            # cleanup queue
            try:
                app.state.sse_queues.remove(q)
            except ValueError:
                pass
    return StreamingResponse(gen(), media_type="text/event-stream")

def broadcast_event(event: dict):
    for q in list(app.state.sse_queues):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass

# --- Safety and topic checks ---
def topic_allowed(text: str):
    text_lower = text.lower()
    # If explicit disallowed patterns found -> refuse
    for patt in DISALLOWED_PATTERNS:
        if re.search(patt, text_lower):
            return False, "Request contains disallowed content (e.g., illegal/medical/financial)."

    # Must contain at least one allowed keyword
    for kw in ALLOWED_TOPICS:
        if kw in text_lower:
            return True, None
    return False, "This chatbot only answers Islamic spiritual health topics: sihr, jinn, ruqyah, waswas, evil eye, dreams."

# --- Simple translator using HF inference API fallback to original if not working ---
def translate(text, source, target):
    # Keep simple; if HF_TOKEN missing, return text
    model_map = {
        ("en", "ms"): "Helsinki-NLP/opus-mt-en-ms",
        ("ms", "en"): "Helsinki-NLP/opus-mt-ms-en",
        ("en", "ar"): "Helsinki-NLP/opus-mt-en-ar",
        ("ar", "en"): "Helsinki-NLP/opus-mt-ar-en",
        ("ms", "ar"): "Helsinki-NLP/opus-mt-ms-ar",
        ("ar", "ms"): "Helsinki-NLP/opus-mt-ar-ms"
    }
    if not HF_TOKEN or (source, target) not in model_map:
        return text
    model = model_map[(source, target)]
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    try:
        r = requests.post(f"https://api-inference.huggingface.co/models/{model}", headers=headers, json={"inputs": text}, timeout=60)
        return r.json()[0]["translation_text"]
    except Exception:
        return text

# --- Ruqyah diagnostic engine (simple rule-based) ---
def ruqyah_assessment(symptoms_text: str):
    txt = symptoms_text.lower()
    score = {"ain":0, "sihr":0, "mass":0, "waswas":0, "hasad":0}
    if any(k in txt for k in ["dream", "nightmare", "mimpi", "كوابيس"]):
        score["sihr"] += 1
        score["mass"] += 1
    if any(k in txt for k in ["menjadi marah", "sudden anger", "غضب", "anger"]):
        score["mass"] += 1
    if any(k in txt for k in ["sudden hatred", "pemisah", "pemisahan", "فراق", "separate"]):
        score["sihr"] += 2
    if "headache" in txt or "sakit kepala" in txt or "صداع" in txt:
        score["mass"] += 1
    if "feel heavy" in txt or "berat" in txt:
        score["mass"] += 2
    if "envy" in txt or "hasad" in txt or "iri" in txt:
        score["hasad"] += 2
    if "whispers" in txt or "waswas" in txt or "وسوسة" in txt:
        score["waswas"] += 3
    # Determine top
    ordered = sorted(score.items(), key=lambda x: x[1], reverse=True)
    top = ordered[0]
    probable = []
    for k,v in score.items():
        if v == ordered[0][1] and v>0:
            probable.append(k)
    if not probable:
        probable = ["unknown"]
    return {"scores": score, "probable": probable}

def generate_ruqyah_plan(category_list):
    # returns a short safe ruqyah plan string
    steps = [
        "1. Istighfar and increase sincere du'a to Allah.",
        "2. Maintain five daily prayers on time and read morning & evening adhkar.",
        "3. Daily recitation of Surah Al-Fatihah, Ayat al-Kursi, last two verses of Al-Baqarah, Al-Ikhlas, Al-Falaq and An-Nas.",
        "4. Avoid visiting fortune-tellers, witch-doctors, and any shirk practices.",
        "5. Seek qualified ruqyah practitioner if severe; consult a trusted scholar/ustadh."
    ]
    if "waswas" in category_list:
        steps.insert(0, "Specific: Increase seeking refuge (A'udhu billahi minash-shaytanir-rajim) and shorten idle solitude; consult a scholar for cognitive support.")
    if "sihr" in category_list or "mass" in category_list:
        steps.insert(0, "Specific: Perform regular ruqyah at home, recite Qur'an over water and drink/spread it, and maintain hospital/medical care if physical symptoms.")
    if "hasad" in category_list:
        steps.insert(0, "Specific: Make regular charity (sadaqah), recite protective supplications, and seek reconciliation if relationships sour.")
    return "\n".join(steps)

# --- Request models ---
class ChatRequest(BaseModel):
    message: str

class TrainRequest(BaseModel):
    question: str
    answer: str

class LoginRequest(BaseModel):
    username: str
    password: str

class AssessmentRequest(BaseModel):
    text: str  # user describes symptoms

# --- Chat endpoint ---
@app.post("/chat")
async def chat(req: ChatRequest):
    user_message = req.message.strip()
    if not user_message:
        return {"response": "Please type a message."}

    try:
        lang = detect(user_message)
    except Exception:
        lang = "en"

    # Safety/topic check
    allowed, reason = topic_allowed(user_message)
    if not allowed:
        return {"response": reason}

    # 1) Check in-memory trained answers first (same language)
    lang_dict = trained_answers.get(lang, {})
    match = difflib.get_close_matches(user_message.lower(), lang_dict.keys(), n=1, cutoff=0.6)
    if match:
        answer = lang_dict[match[0]]
    else:
        # 2) Query HF model if available, with strict system prompt
        system_prompt = (
            "You are an Islamic assistant: only answer questions about spiritual sickness, jinn, sihr (magic), ruqyah, waswas, evil eye, and dreams interpretation. "
            "Answer strictly according to Quran and authentic Sunnah where applicable. Do not provide medical, financial, political, or illegal instructions. "
            "If uncertain, say 'Wallahu A'lam' and advise to consult a qualified scholar or ruqyah practitioner. Keep answers concise and practical."
        )
        eng_msg = user_message if lang == "en" else translate(user_message, lang, "en")
        if client:
            try:
                completion = client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": eng_msg},
                    ],
                )
                reply = completion.choices[0].message["content"]
            except Exception:
                reply = "Maaf, perkhidmatan respons tidak boleh dihubungi sekarang. Sila cuba lagi kemudian."
        else:
            # Fallback: conservative default answer
            reply = (
                "Wallahu A'lam. I can only provide general Islamic guidance. "
                "Please provide more details or consult a qualified scholar/ruqyah practitioner."
            )
        # Translate reply back if necessary
        if lang != "en":
            reply = translate(reply, "en", lang)
        answer = reply

    # Save chat to persistent storage
    chats = load_json(CHAT_FILE).get("chats", [])
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "message": user_message,
        "response": answer,
        "lang": lang
    }
    chats.append(entry)
    save_json(CHAT_FILE, {"chats": chats})

    # Broadcast to SSE (admin live feed)
    broadcast_event({"type":"chat", "data": entry})

    # Save into training data automatically (optional) - keep but flagged so admin can curate
    # We append to training_data.json as assistant-generated example flagged with auto:true
    training = load_json(TRAIN_FILE).get("training_data", [])
    training.append({"question": user_message, "answer": answer, "lang": lang, "auto": True, "timestamp": datetime.utcnow().isoformat()})
    save_json(TRAIN_FILE, {"training_data": training})
    # refresh in-memory
    load_training_to_memory()

    return {"response": answer}

# --- Ruqyah assessment endpoint ---
@app.post("/assess")
async def assess(req: AssessmentRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Please describe symptoms.")
    # Topic check (still require relevant)
    allowed, reason = topic_allowed(text)
    if not allowed:
        return {"response": reason}

    try:
        lang = detect(text)
    except Exception:
        lang = "en"

    assessment = ruqyah_assessment(text)
    ruqyah_plan = generate_ruqyah_plan(assessment["probable"])

    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": text,
        "lang": lang,
        "assessment": assessment,
        "ruqyah_plan": ruqyah_plan
    }

    # Save to chat_history as an assessment record
    chats = load_json(CHAT_FILE).get("chats", [])
    chats.append({"type": "assessment", **result})
    save_json(CHAT_FILE, {"chats": chats})

    # Broadcast to SSE
    broadcast_event({"type":"assessment", "data": result})

    return {"result": result}

# --- Training endpoint (admin can call) ---
@app.post("/train")
async def train(req: TrainRequest, admin: str = Depends(admin_required)):
    question = req.question.strip()
    answer = req.answer.strip()
    if not question or not answer:
        raise HTTPException(status_code=400, detail="Please provide both question and answer.")

    try:
        lang = detect(question)
    except Exception:
        lang = "en"

    data = load_json(TRAIN_FILE).get("training_data", [])
    updated = False
    for item in data:
        if item.get("lang") == lang and item["question"].lower() == question.lower():
            item["answer"] = answer
            item["auto"] = False
            updated = True
            break
    if not updated:
        data.append({"question": question, "answer": answer, "lang": lang, "auto": False, "timestamp": datetime.utcnow().isoformat()})
    save_json(TRAIN_FILE, {"training_data": data})
    load_training_to_memory()

    # Broadcast to SSE so admin dashboard updates
    broadcast_event({"type":"train_update", "data": {"question": question, "lang": lang, "updated": updated}})

    return {"message": f"{'Updated' if updated else 'Added'} {lang.upper()} training data successfully."}

# --- Get training data (admin) ---
@app.get("/get-training-data")
async def get_training_data(admin: str = Depends(admin_required)):
    data = load_json(TRAIN_FILE).get("training_data", [])
    return {"training_data": data}

# --- Get chat history (admin) ---
@app.get("/get-chats")
async def get_chats(admin: str = Depends(admin_required)):
    data = load_json(CHAT_FILE).get("chats", [])
    return {"chats": data}

# --- Admin analytics (protected) ---
@app.get("/admin-stats")
async def admin_stats(key: str = Query(...)):
    # kept for backwards compatibility (key param)
    if key != os.getenv("ADMIN_KEY", "mc250132689"):
        return {"error": "Unauthorized"}

    training = load_json(TRAIN_FILE).get("training_data", [])
    chats = load_json(CHAT_FILE).get("chats", [])

    total_records = len(training)
    lang_count = Counter(item.get("lang", "unknown") for item in training)

    avg_q_len = round(sum(len(item["question"]) for item in training) / total_records, 1) if total_records else 0
    avg_a_len = round(sum(len(item["answer"]) for item in training) / total_records, 1) if total_records else 0

    # Chat analytics
    total_chats = len(chats)
    types = Counter(c.get("type", "chat") for c in chats)
    # basic keyword frequency from questions
    kw_counter = Counter()
    for c in chats:
        txt = (c.get("message") or c.get("input") or "").lower()
        for word in re.findall(r"\w+", txt):
            if len(word) > 3:
                kw_counter[word] += 1
    top_keywords = kw_counter.most_common(10)

    return {
        "total_records": total_records,
        "languages": dict(lang_count),
        "avg_question_length": avg_q_len,
        "avg_answer_length": avg_a_len,
        "total_chats": total_chats,
        "chat_types": dict(types),
        "top_keywords": top_keywords
    }

# --- Simple login endpoint (returns JWT) ---
@app.post("/login")
async def login(req: LoginRequest):
    users = load_users()
    for u in users:
        if u["username"] == req.username:
            if bcrypt.verify(req.password, u["password"]):
                token = create_access_token({"sub": req.username})
                return {"access_token": token, "token_type": "bearer"}
            break
    raise HTTPException(status_code=401, detail="Invalid credentials")

# --- Optional: create admin (not public) ---
@app.post("/create-admin")
async def create_admin(req: LoginRequest, admin: str = Depends(admin_required)):
    hashed = bcrypt.hash(req.password)
    save_user(req.username, hashed)
    return {"message": "Admin user created/updated."}
