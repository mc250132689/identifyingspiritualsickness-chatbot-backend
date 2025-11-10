from fastapi import FastAPI, Query
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

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Hugging Face GPT-OSS client (Groq) ===
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(api_key=HF_TOKEN)  # DO NOT set provider here

DATA_FILE = "training_data.json"
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump({"training_data": []}, f, ensure_ascii=False, indent=2)

trained_answers = {}

# Normalizer for better matching
_non_alnum_re = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_multi_space_re = re.compile(r"\s+", flags=re.UNICODE)

def normalize_text(s: str) -> str:
    s = s or ""
    s = s.lower()
    s = _non_alnum_re.sub(" ", s)
    s = _multi_space_re.sub(" ", s).strip()
    return s

def load_data():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data_list = json.load(f).get("training_data", [])
    global trained_answers
    trained_answers = {}
    for item in data_list:
        lang = item.get("lang", "en")
        q = item.get("question", "").strip()
        a = item.get("answer", "")
        if lang not in trained_answers:
            trained_answers[lang] = {}
        # store both raw and normalized key for matching
        trained_answers[lang][q.lower()] = {
            "answer": a,
            "norm": normalize_text(q)
        }
    return data_list

def save_data(data_list):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump({"training_data": data_list}, f, ensure_ascii=False, indent=2)

load_data()

def translate(text, source, target):
    model_map = {
        ("en", "ms"): "Helsinki-NLP/opus-mt-en-ms",
        ("ms", "en"): "Helsinki-NLP/opus-mt-ms-en",
        ("en", "ar"): "Helsinki-NLP/opus-mt-en-ar",
        ("ar", "en"): "Helsinki-NLP/opus-mt-ar-en",
        ("ms", "ar"): "Helsinki-NLP/opus-mt-ms-ar",
        ("ar", "ms"): "Helsinki-NLP/opus-mt-ar-ms"
    }
    if (source, target) not in model_map:
        return text
    model = model_map[(source, target)]
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    try:
        r = requests.post(
            f"https://api-inference.huggingface.co/models/{model}",
            headers=headers,
            json={"inputs": text},
            timeout=60,
        )
        return r.json()[0]["translation_text"]
    except:
        return text

class ChatRequest(BaseModel):
    message: str

class TrainRequest(BaseModel):
    question: str
    answer: str

@app.post("/chat")
async def chat(req: ChatRequest):
    user_message = req.message.strip()
    if not user_message:
        return {"response": "Please type a message."}

    try:
        lang = detect(user_message)
    except:
        lang = "en"

    # Try trained answers first with improved normalization
    lang_dict = trained_answers.get(lang, {})
    norm_user = normalize_text(user_message)

    best_match = None
    best_score = 0.0

    for original_q, meta in lang_dict.items():
        score = difflib.SequenceMatcher(None, norm_user, meta["norm"]).ratio()
        if score > best_score:
            best_score = score
            best_match = original_q

    # threshold tuned to 0.70 for higher precision; adjust if you want more recall
    if best_score >= 0.70 and best_match:
        return {"response": lang_dict[best_match]["answer"]}

    # fallback translate to english before calling model
    eng_msg = user_message if lang == "en" else translate(user_message, lang, "en")

    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b:groq",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an Islamic knowledge assistant specializing in spiritual sickness, "
                    "jin possession, sihr, ruqyah and Islamic dream interpretation. "
                    "Your answers MUST only reference Quran, Sahih Hadith, and valid ruqyah practices. "
                    "Never provide mystical, speculative, or cultural superstition advice."
                ),
            },
            {"role": "user", "content": eng_msg}
        ],
    )

    reply = completion.choices[0].message["content"]
    if lang != "en":
        reply = translate(reply, "en", lang)

    data = load_data()
    data.append({"question": user_message, "answer": reply, "lang": lang})
    save_data(data)

    # update in-memory trained_answers
    if lang not in trained_answers:
        trained_answers[lang] = {}
    trained_answers[lang][user_message.lower()] = {"answer": reply, "norm": normalize_text(user_message)}

    return {"response": reply}

@app.post("/train")
async def train(req: TrainRequest):
    question = req.question.strip()
    answer = req.answer.strip()
    if not question or not answer:
        return {"message": "Please provide both question and answer."}

    try:
        lang = detect(question)
    except:
        lang = "en"

    data = load_data()
    updated = False
    for item in data:
        if item.get("lang", "en") == lang and item.get("question", "").strip().lower() == question.lower():
            item["answer"] = answer
            updated = True
            break
    if not updated:
        data.append({"question": question, "answer": answer, "lang": lang})
    save_data(data)

    # update in-memory store
    if lang not in trained_answers:
        trained_answers[lang] = {}
    trained_answers[lang][question.lower()] = {"answer": answer, "norm": normalize_text(question)}

    return {"message": f"{'Updated' if updated else 'Added'} {lang.upper()} training data successfully."}

@app.get("/get-training-data")
async def get_training_data():
    return {"training_data": load_data()}

ADMIN_KEY = os.getenv("ADMIN_KEY", "mc250132689")

@app.get("/admin-stats")
async def admin_stats(key: str = Query(...)):
    if key != ADMIN_KEY:
        return {"error": "Unauthorized"}

    data = load_data()
    total = len(data)
    lang_count = Counter(item.get("lang", "en") for item in data)

    avg_q = round(sum(len(i.get("question", "")) for i in data) / total, 1) if total else 0
    avg_a = round(sum(len(i.get("answer", "")) for i in data) / total, 1) if total else 0

    return {
        "total_records": total,
        "languages": dict(lang_count),
        "avg_question_length": avg_q,
        "avg_answer_length": avg_a
    }

# --- New endpoint: hadith search within stored training data (simple keyword scan) ---
@app.get("/hadith-search")
async def hadith_search(q: str = Query(...)):
    """
    Search saved training answers for hadith references or matches.
    This scans stored training data and returns records whose answer or question contains the query or hadith keywords.
    """
    qnorm = normalize_text(q)
    data = load_data()
    keywords = ["bukhari", "muslim", "sahih", "riyad", "tirmidhi", "abu dawood", "nasai", "ibn majah", "hadith", "riyadh", "sahih al-bukhari", "sahih muslim"]

    results = []
    for item in data:
        combined = f"{item.get('question','')} {item.get('answer','')}"
        if qnorm in normalize_text(combined) or any(kw in combined.lower() for kw in keywords):
            results.append(item)

    return {"query": q, "count": len(results), "results": results}
