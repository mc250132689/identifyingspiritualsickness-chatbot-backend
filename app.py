# app.py (Part 1 of 3) - Postgres-ready, upgraded sentiment/emotion (split into 3 parts)
# ---------------------------
# Imports & configuration
# ---------------------------
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

# Optional heavy libs for local model inference - lazy import handled below
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    import numpy as np
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

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
# Models & mapping globals (lazy loaded)
# ---------------------------
# Preferred model names (can be changed later)
MULTILINGUAL_SENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

sent_tokenizer = None
sent_model = None
sent_labels = ["Negative", "Neutral", "Positive"]

emo_tokenizer = None
emo_model = None
emo_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

islamic_map = {
    "fear": "Khawf (anxiety / was-was / fear of harm)",
    "sadness": "Huzn (sorrow / spiritual distress)",
    "joy": "Farah (gratitude / relief / happiness)",
    "anger": "Ghadab (anger / frustration)",
    "disgust": "Karahah (repulsion / dislike)",
    "surprise": "Tafakkur (reflection / amazement)",
    "neutral": "Sakinah (calmness / stability)"
}

# ---------------------------
# PostgreSQL helpers
# ---------------------------
def build_dsn_from_env():
    """Return a DSN string for psycopg2 from env vars, or DATABASE_URL if provided."""
    if DATABASE_URL:
        return DATABASE_URL
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
        dsn_parts.append(f"channel_binding={CHANNEL_BINDING}")
    return " ".join(dsn_parts)

def get_db():
    """
    Returns a new psycopg2 connection. Caller must close().
    Uses DATABASE_URL if provided, otherwise builds DSN from DB_* env vars.
    """
    dsn = build_dsn_from_env()
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
try:
    init_db()
except Exception as e:
    print("[INIT_DB] error:", e)

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
# Sentiment/emotion lazy loader
# ---------------------------
def load_models_if_needed():
    global sent_tokenizer, sent_model, emo_tokenizer, emo_model
    if not TRANSFORMERS_AVAILABLE:
        return False
    try:
        if sent_tokenizer is None or sent_model is None:
            sent_tokenizer = AutoTokenizer.from_pretrained(MULTILINGUAL_SENT_MODEL)
            sent_model = AutoModelForSequenceClassification.from_pretrained(MULTILINGUAL_SENT_MODEL)
        if emo_tokenizer is None or emo_model is None:
            emo_tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL)
            emo_model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL)
        return True
    except Exception as e:
        print("[MODEL LOAD] error:", e)
        return False

async def hf_sentiment(text: str):
    """
    Upgraded multilingual sentiment + emotion + Islamic mapping
    Returns a dict with sentiment, emotion, islamic_emotion, confidence
    Falls back to HuggingFace InferenceClient if local transformers not available
    """
    if not text or text.strip() == "":
        return {
            "sentiment": "Neutral",
            "emotion": "neutral",
            "islamic_emotion": islamic_map.get("neutral"),
            "confidence": 0.0
        }

    # Try local transformers first (if installed)
    if TRANSFORMERS_AVAILABLE and load_models_if_needed():
        try:
            sent_inputs = sent_tokenizer(text, return_tensors="pt", truncation=True)
            sent_outputs = sent_model(**sent_inputs)
            sent_scores = torch.softmax(sent_outputs.logits, dim=1).detach().numpy()[0]
            sent_idx = int(np.argmax(sent_scores))
            sentiment = sent_labels[sent_idx]
            sent_conf = float(sent_scores[sent_idx])

            emo_inputs = emo_tokenizer(text, return_tensors="pt", truncation=True)
            emo_outputs = emo_model(**emo_inputs)
            emo_scores = torch.softmax(emo_outputs.logits, dim=1).detach().numpy()[0]
            emo_idx = int(np.argmax(emo_scores))
            emotion_raw = emo_labels[emo_idx]
            islamic_emotion = islamic_map.get(emotion_raw, islamic_map.get("neutral"))

            return {
                "sentiment": sentiment,
                "emotion": emotion_raw,
                "islamic_emotion": islamic_emotion,
                "confidence": round(sent_conf, 4)
            }
        except Exception as e:
            print("[LOCAL SENT] error:", e)
            # fall through to inference client

    # Fallback to InferenceClient (if configured)
    if client:
        try:
            sres = await asyncio.to_thread(lambda: client.text_classification(model=MULTILINGUAL_SENT_MODEL, inputs=text))
            if isinstance(sres, list) and len(sres) > 0:
                s_lbl = sres[0].get("label")
                s_score = float(sres[0].get("score", 0.0))
                lbl = str(s_lbl).lower()
                if "neg" in lbl or "0" in lbl:
                    sent = "Negative"
                elif "neu" in lbl or "1" in lbl:
                    sent = "Neutral"
                else:
                    sent = "Positive"
            else:
                sent = "Neutral"
                s_score = 0.0

            eres = await asyncio.to_thread(lambda: client.text_classification(model=EMOTION_MODEL, inputs=text))
            if isinstance(eres, list) and len(eres) > 0:
                e_lbl = str(eres[0].get("label", "neutral")).lower()
                emotion_raw = e_lbl if e_lbl in emo_labels else ("neutral" if "neu" in e_lbl else e_lbl)
            else:
                emotion_raw = "neutral"

            islamic_emotion = islamic_map.get(emotion_raw, islamic_map.get("neutral"))

            return {
                "sentiment": sent,
                "emotion": emotion_raw,
                "islamic_emotion": islamic_emotion,
                "confidence": round(float(s_score), 4)
            }
        except Exception as e:
            print("[HF INFERENCE] error:", e)

    # last resort fallback
    return {
        "sentiment": "Neutral",
        "emotion": "neutral",
        "islamic_emotion": islamic_map.get("neutral"),
        "confidence": 0.0
    }

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
        except Exception as e:
            print("[CHAT FALLBACK] error:", e)
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

# app.py (Part 2 of 3) - training, feedback, guidance, hadith-search, admin-stats

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

# ---------------------------
# Feedback endpoints (with sentiment/emotion)
# ---------------------------
@app.post("/submit-feedback")
async def submit_feedback(item: FeedbackItem):
    # collect comment text (prefer explicit comments field; fallback to q14)
    comment_text = (item.comments or "").strip()
    if not comment_text:
        # use q14 if present in object (some frontends send q14 as comments)
        try:
            # pydantic model doesn't include q12..q14 as attributes other than comments,
            # but some payloads might include them; check raw dict via item.dict()
            raw = item.dict()
            comment_text = (raw.get("q14") or raw.get("comments") or "").strip()
        except Exception:
            comment_text = ""

    # compute sentiment/emotion (async)
    try:
        sent_obj = await hf_sentiment(comment_text)
    except Exception as e:
        print("[SENTIMENT] error:", e)
        sent_obj = {
            "sentiment": "Neutral",
            "emotion": "neutral",
            "islamic_emotion": islamic_map.get("neutral"),
            "confidence": 0.0
        }

    payload = item.dict()
    # attach full sentiment/emotion info
    payload["sentiment"] = sent_obj.get("sentiment")
    payload["emotion"] = sent_obj.get("emotion")
    payload["islamic_emotion"] = sent_obj.get("islamic_emotion")
    payload["sentiment_confidence"] = sent_obj.get("confidence")

    conn = get_db()
    cur = conn.cursor()
    cur.execute("INSERT INTO feedback (data) VALUES (%s)", (json.dumps(payload),))
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
    out = []
    for r in rows:
        d = r["data"]
        # data may be stored as JSON string or as JSON object depending on psycopg2/cfg
        if isinstance(d, str):
            try:
                d = json.loads(d)
            except Exception:
                try:
                    # last resort: attempt eval (not ideal), but safer to keep original string
                    d = {"raw": d}
                except:
                    d = {"raw": d}
        out.append(d)
    return {"feedback": out}

# ---------------------------
# Guidance endpoint (uses symptom classifier + heuristics)
# ---------------------------
@app.post("/guidance")
async def guidance(req: GuidanceRequest):
    symptoms = (req.symptoms or "") + " " + (req.details or "")
    s = normalize_text(symptoms)
    label, score = "none", 0.0
    lang = "en"
    if symptoms and symptoms.strip():
        try:
            lang = detect(symptoms)
        except:
            lang = "en"
        text_for_model = symptoms
        MODEL_ID = "your-hf-symptoms-classifier"
        label, score = await hf_symptom_classify(text_for_model, MODEL_ID)

    threshold = 0.6
    if score < threshold or not symptoms.strip():
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

    # feedback stats
    cur.execute("SELECT * FROM feedback ORDER BY id ASC")
    fb = cur.fetchall()
    feedback_count = len(fb)

    # sentiment and islamic emotion summary
    sentiment_counts = Counter()
    islamic_counts = Counter()
    senti_examples = []
    for row in fb:
        d = row["data"]
        if isinstance(d, str):
            try:
                d = json.loads(d)
            except:
                d = {}
        sent = d.get("sentiment") or d.get("sentiment_label") or "Neutral"
        # normalize
        sent_norm = sent.capitalize() if isinstance(sent, str) else "Neutral"
        sentiment_counts[sent_norm] += 1

        islamic = d.get("islamic_emotion") or d.get("islamic") or None
        if islamic:
            islamic_counts[islamic] += 1

        if len(senti_examples) < 5:
            senti_examples.append(d)

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
        "feedback_count": feedback_count,
        "sentiment_summary": dict(sentiment_counts),
        "islamic_emotion_summary": dict(islamic_counts),
        "sentiment_examples": senti_examples
    }

# app.py (Part 3 of 3) - utilities, HF helpers, normalization, Islamic emotion map, keep-alive

# ---------------------------
# Utilities
# ---------------------------
def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_hadith_refs(text: str):
    # crude pattern for hadith references, customize as needed
    pattern = r"(?:Hadith|Riwayat)\s*[0-9]+"
    return re.findall(pattern, text)

def get_db():
    import psycopg2
    import psycopg2.extras
    conn = psycopg2.connect(DB_URL)
    return conn

# ---------------------------
# Hugging Face helpers (async)
# ---------------------------
async def hf_sentiment(text: str):
    # detect multilingual sentiment + islamic emotion
    if not text.strip():
        return {"sentiment":"Neutral","emotion":"neutral","islamic_emotion":"neutral","confidence":0.0}
    try:
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        payload = {"inputs": text}
        async with aiohttp.ClientSession() as session:
            async with session.post(HF_SENTIMENT_MODEL, headers=headers, json=payload) as resp:
                r = await resp.json()
        label = r[0]["label"] if isinstance(r, list) and "label" in r[0] else "Neutral"
        score = float(r[0]["score"]) if isinstance(r, list) and "score" in r[0] else 0.0
        # Map label to emotion
        emotion = label.lower()
        islamic_emotion = islamic_map.get(emotion, "neutral")
        sentiment = "Positive" if emotion in ["joy","happy","satisfied"] else "Negative" if emotion in ["anger","fear","sad","disgust"] else "Neutral"
        return {"sentiment":sentiment,"emotion":emotion,"islamic_emotion":islamic_emotion,"confidence":score}
    except Exception as e:
        print("[HF SENTIMENT] error:", e)
        return {"sentiment":"Neutral","emotion":"neutral","islamic_emotion":"neutral","confidence":0.0}

async def hf_symptom_classify(text: str, model_id: str):
    # crude HF symptom classifier (jin, sihr, ruqyah)
    if not text.strip():
        return "none", 0.0
    try:
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        payload = {"inputs": text}
        async with aiohttp.ClientSession() as session:
            async with session.post(f"https://api-inference.huggingface.co/models/{model_id}", headers=headers, json=payload) as resp:
                r = await resp.json()
        # crude extraction
        if isinstance(r, dict) and "label" in r and "score" in r:
            return r["label"], float(r["score"])
        return "none", 0.0
    except Exception as e:
        print("[HF SYMPTOM] error:", e)
        return "none", 0.0

# ---------------------------
# Islamic emotion mapping
# ---------------------------
islamic_map = {
    "neutral": "Sakina",        # calm
    "joy": "Farah",             # happiness
    "happy": "Farah",
    "satisfied": "Rida",        # contentment
    "sad": "Huzn",              # sadness
    "fear": "Khawf",            # fear
    "anger": "Ghadab",          # anger
    "disgust": "Qayr",          # displeasure
    "surprise": "Taajjub"
}

# ---------------------------
# Startup and background tasks
# ---------------------------
async def keep_awake():
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                await session.get(SERVER_URL)
            print("[KEEP-AWAKE] pinged server successfully.")
        except Exception as e:
            print("[KEEP-AWAKE] failed:", e)
        await asyncio.sleep(300)  # every 5 min

@app.on_event("startup")
async def startup_event():
    print("[STARTUP] server starting...")
    asyncio.create_task(keep_awake())
    # preload training data
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT question, answer, lang FROM training_data")
        rows = cur.fetchall()
        for q,a,l in rows:
            if l not in trained_answers: trained_answers[l] = {}
            trained_answers[l][q.lower()] = {"answer": a, "norm": normalize_text(q)}
        cur.close()
        conn.close()
        print(f"[STARTUP] loaded {len(rows)} training records.")
    except Exception as e:
        print("[STARTUP] failed to load training data:", e)

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
