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

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Hugging Face GPT-OSS client (Groq) ---
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(api_key=HF_TOKEN)

# --- Ensure data folders exist ---
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
DATA_FILE = os.path.join(DATA_DIR, "training_data.json")
FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback.json")

# --- Initialize files if missing ---
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump({"training_data": []}, f, ensure_ascii=False, indent=2)

if not os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump({"feedback": []}, f, ensure_ascii=False, indent=2)

trained_answers = {}

# --- Text normalization ---
_non_alnum_re = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_multi_space_re = re.compile(r"\s+", flags=re.UNICODE)
def normalize_text(s: str) -> str:
    s = s or ""
    s = s.lower()
    s = _non_alnum_re.sub(" ", s)
    s = _multi_space_re.sub(" ", s).strip()
    return s

def atomic_write_json(path: str, data):
    dirn = os.path.dirname(path)
    os.makedirs(dirn, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dirn, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        shutil.move(tmp_path, path)
    except Exception:
        try: os.remove(tmp_path)
        except Exception: pass
        raise

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
        trained_answers[lang][q.lower()] = {"answer": a, "norm": normalize_text(q)}
    return data_list

def save_data(data_list):
    atomic_write_json(DATA_FILE, {"training_data": data_list})

load_data()

# --- Translation utility ---
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
    except Exception:
        return text

# --- Hadith extraction ---
HADITH_KEYWORDS = [
    "bukhari","muslim","sahih","riyad","tirmidhi","abu dawood","nasai","ibn majah",
    "hadith","riyadh","sahih al-bukhari","sahih muslim","malik"
]
HADITH_REF_RE = re.compile(r"(bukhari|sahih al-bukhari|sahih muslim|muslim|tirmidhi|abu dawood|nasai|ibn majah|riyad)\b", flags=re.IGNORECASE)
def extract_hadith_refs(text: str) -> List[str]:
    refs = set()
    if not text: return []
    for m in HADITH_REF_RE.finditer(text):
        refs.add(m.group(0).strip())
    lowered = text.lower()
    for kw in HADITH_KEYWORDS:
        if kw in lowered: refs.add(kw)
    return list(refs)

# --- Async Hugging Face symptom classifier ---
async def hf_symptom_classify(text: str, model_id: str):
    try:
        result = await asyncio.to_thread(lambda: client.text_classification(model=model_id, inputs=text))
        if result:
            return result[0]["label"].lower(), result[0]["score"]
    except Exception:
        pass
    return "none", 0.0

# --- Pydantic models ---
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

# --- Chat endpoint ---
@app.post("/chat")
async def chat(req: ChatRequest):
    user_message = req.message.strip()
    if not user_message: return {"response": "Please type a message."}

    try: lang = detect(user_message)
    except Exception: lang = "en"

    norm_user = normalize_text(user_message)
    lang_dict = trained_answers.get(lang, {})

    # --- Step 1: trained answers ---
    best_match, best_score = None, 0.0
    for original_q, meta in lang_dict.items():
        score = difflib.SequenceMatcher(None, norm_user, meta["norm"]).ratio()
        if score > best_score:
            best_score = score
            best_match = original_q
    if best_score >= 0.70 and best_match:
        return {"response": lang_dict[best_match]["answer"]}

    # --- Step 2: symptom detection ---
    symptom_keywords = ["voices","see","visions","insomnia","nightmares","dizziness","palpitation","itching","sudden pain","fear","not myself"]
    is_symptoms = any(k in norm_user for k in symptom_keywords)
    label, score = "none", 0.0
    MODEL_ID = "your-hf-symptoms-classifier"

    if is_symptoms:
        text_for_model = user_message if lang=="en" else translate(user_message, lang, "en")
        label, score = await hf_symptom_classify(text_for_model, MODEL_ID)

    if score >= 0.6:
        guidance_labels = {"jin":"jin involvement","sihir":"possible sihr","ruqyah":"symptoms treated with ruqyah"}
        suggestions = [f"Detected symptoms suggest {guidance_labels.get(label,'general ruqyah guidance')}. Consider seeking a qualified ruqyah practitioner and medical check-up."]
        return {"response":"⚠️ Symptom-based guidance provided.","model_label":label,"model_confidence":score,"suggestions":suggestions}

    # --- Step 3: GPT-OSS fallback ---
    eng_msg = user_message if lang=="en" else translate(user_message, lang, "en")
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b:groq",
        messages=[
            {"role":"system","content":"You are an Islamic knowledge assistant specializing in spiritual sickness, jin possession, sihr, ruqyah and Islamic dream interpretation. Answers MUST reference Quran, Sahih Hadith, valid ruqyah practices only."},
            {"role":"user","content":eng_msg}
        ]
    )
    reply = completion.choices[0].message["content"]
    if lang!="en": reply = translate(reply,"en",lang)

    hadith_refs = extract_hadith_refs(reply)
    data = load_data()
    entry = {"question":user_message,"answer":reply,"lang":lang,"hadith_refs":hadith_refs}
    data.append(entry)
    save_data(data)
    if lang not in trained_answers: trained_answers[lang]={}
    trained_answers[lang][user_message.lower()]={"answer":reply,"norm":norm_user}

    return {"response":reply,"hadith_refs":hadith_refs}

# --- Training endpoints ---
@app.post("/train")
async def train(req: TrainRequest):
    question, answer = req.question.strip(), req.answer.strip()
    if not question or not answer: return {"message": "Please provide both question and answer."}
    try: lang = detect(question)
    except Exception: lang = "en"

    data = load_data()
    updated = False
    for item in data:
        if item.get("lang","en")==lang and item.get("question","").strip().lower()==question.lower():
            item["answer"] = answer
            item["hadith_refs"] = extract_hadith_refs(answer)
            updated = True
            break
    if not updated:
        data.append({"question": question, "answer": answer, "lang": lang, "hadith_refs": extract_hadith_refs(answer)})
    save_data(data)
    if lang not in trained_answers: trained_answers[lang]={}
    trained_answers[lang][question.lower()]={"answer": answer, "norm": normalize_text(question)}
    return {"message": f"{'Updated' if updated else 'Added'} {lang.upper()} training data successfully."}

@app.get("/get-training-data")
async def get_training_data():
    return {"training_data": load_data()}

@app.get("/export-training-data")
async def export_training_data():
    return {"training_data": load_data()}

@app.post("/import-training-data")
async def import_training_data(payload: dict):
    incoming = payload.get("training_data")
    if not isinstance(incoming, list):
        raise HTTPException(status_code=400, detail="Invalid payload; expected training_data list")
    existing = load_data()
    existing_norms = {normalize_text(i.get("question","")): i for i in existing}
    merged = existing[:]
    added = updated = 0
    for item in incoming:
        q = item.get("question","").strip()
        if not q: continue
        norm = normalize_text(q)
        if norm in existing_norms:
            ex_item = existing_norms[norm]
            if ex_item.get("answer","").strip() != item.get("answer","").strip():
                ex_item["answer"] = item.get("answer","").strip()
                ex_item["hadith_refs"] = extract_hadith_refs(ex_item["answer"])
                updated +=1
        else:
            merged.append({
                "question": q,
                "answer": item.get("answer","").strip(),
                "lang": item.get("lang","en"),
                "hadith_refs": extract_hadith_refs(item.get("answer",""))
            })
            added +=1
    save_data(merged)
    return {"message": f"Imported. Added: {added}, Updated: {updated}", "added": added, "updated": updated}

# --- Admin & stats ---
ADMIN_KEY = os.getenv("ADMIN_KEY", "mc250132689")
@app.get("/admin-stats")
async def admin_stats(key: str = Query(...)):
    if key != ADMIN_KEY: return {"error": "Unauthorized"}
    data = load_data()
    total = len(data)
    lang_count = Counter(item.get("lang","en") for item in data)
    avg_q = round(sum(len(i.get("question","")) for i in data)/total,1) if total else 0
    avg_a = round(sum(len(i.get("answer","")) for i in data)/total,1) if total else 0
    hadith_count = sum(1 for i in data if i.get("hadith_refs"))
    hadith_examples = [i for i in data if i.get("hadith_refs")][:5]
    q_counter = Counter(normalize_text(i.get("question","")) for i in data)
    top_questions = [q for q,_ in q_counter.most_common(10)]
    recent = data[-10:] if total else []
    try:
        with open(FEEDBACK_FILE,"r",encoding="utf-8") as f: fb=json.load(f).get("feedback",[])
    except Exception: fb=[]
    feedback_count = len(fb)
    return {"total_records": total, "languages": dict(lang_count), "avg_question_length": avg_q, "avg_answer_length": avg_a, "hadith_count": hadith_count, "hadith_examples": hadith_examples, "top_questions": top_questions, "recent_records": recent, "feedback_count": feedback_count}

@app.get("/hadith-search")
async def hadith_search(q: str = Query(...)):
    qnorm = normalize_text(q)
    data = load_data()
    keywords = HADITH_KEYWORDS
    results=[]
    for item in data:
        combined = f"{item.get('question','')} {item.get('answer','')}"
        if qnorm in normalize_text(combined) or any(kw in combined.lower() for kw in keywords):
            results.append(item)
    return {"query": q, "count": len(results), "results": results}

# --- Guidance endpoint (async HF integrated) ---
@app.post("/guidance")
async def guidance(req: GuidanceRequest):
    symptoms = (req.symptoms or "") + " " + (req.details or "")
    s = normalize_text(symptoms)
    label, score = "none",0.0
    lang="en"
    if symptoms:
        try: lang = detect(symptoms)
        except Exception: lang="en"
        text_for_model = symptoms if lang=="en" else translate(symptoms, lang, "en")
        MODEL_ID = "your-hf-symptoms-classifier"
        label, score = await hf_symptom_classify(text_for_model, MODEL_ID)

    threshold=0.6
    if score<threshold or not symptoms:
        keywords_jin = ["voices","hear voices","see","seeing","visions","speaking","possession","control me","not myself","sudden change"]
        keywords_sihir = ["sudden illness","sudden poverty","bad luck","marriage problem","family problem","sudden hatred","sudden fear"]
        keywords_ruqyah = ["insomnia","nightmares","sleepless","dizziness","palpitation","weird smell","itching","sudden pain"]
        matched_jin = any(k in s for k in keywords_jin)
        matched_sihir = any(k in s for k in keywords_sihir)
        matched_ruqyah = any(k in s for k in keywords_ruqyah)
    else:
        matched_jin = label=="jin"
        matched_sihir = label=="sihir"
        matched_ruqyah = label=="ruqyah"

    suggestions=[]
    severity="low"
    if matched_jin:
        severity="high"
        suggestions.append("Signs suggest possible jin involvement. Seek qualified ruqyah practitioner & medical opinion.")
    if matched_sihir:
        severity="medium" if severity!="high" else severity
        suggestions.append("Signs suggest possible sihr. Document, seek ruqyah & verify with scholars.")
    if matched_ruqyah:
        severity="medium"
        suggestions.append("Symptoms match ruqyah-treated cases. Follow Quranic ruqyah steps.")

    if not (matched_jin or matched_sihir or matched_ruqyah):
        suggestions.append("Symptoms not strongly suggestive of jin/sihr. Seek medical check, consult scholar.")
        severity="low"

    steps=[
        "1) Seek immediate medical check-up.",
        "2) Consult qualified ruqyah practitioner using Quran & authentic hadith.",
        "3) Maintain adhkar, prayer, Surah Al-Fatihah, Ayat al-Kursi, last two surahs.",
        "4) Avoid unqualified practitioners or un-Islamic practices."
    ]

    return {"severity":severity,"matched_jin":matched_jin,"matched_sihir":matched_sihir,"matched_ruqyah":matched_ruqyah,"suggestions":suggestions,"recommended_steps":steps,"model_label":label,"model_confidence":score,"language_detected":lang}

# --- Feedback endpoints ---
@app.post("/submit-feedback")
async def submit_feedback(item: FeedbackItem):
    try:
        with open(FEEDBACK_FILE,"r",encoding="utf-8") as f: fb=json.load(f).get("feedback",[])
    except Exception: fb=[]
    fb.append(item.dict())
    atomic_write_json(FEEDBACK_FILE, {"feedback": fb})
    return {"message": "Feedback submitted. Jazakallah khair."}

@app.get("/export-feedback")
async def export_feedback(key: str = Query(None)):
    if key and key==ADMIN_KEY:
        try:
            with open(FEEDBACK_FILE,"r",encoding="utf-8") as f: fb=json.load(f)
        except Exception: fb={"feedback": []}
        return fb
    else:
        raise HTTPException(status_code=403, detail="Unauthorized")

@app.get("/get-feedback")
def get_feedback():
    if not os.path.exists(FEEDBACK_FILE): return []
    try:
        with open(FEEDBACK_FILE,"r",encoding="utf-8") as f:
            return json.load(f).get("feedback",[])
    except Exception:
        return []

# --- Health / Ping endpoints ---
@app.get("/health")
async def health():
    """
    Simple health check endpoint for monitoring.
    Returns ok if the server is running.
    """
    return {"status": "ok"}

@app.get("/ping")
async def ping():
    """
    Lightweight ping endpoint to check if the server is awake.
    Can be used by frontend to wake up free-tier hosting servers.
    """
    return {"status": "ok", "message": "Server is awake"}

