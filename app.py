from fastapi import FastAPI, Query, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from langdetect import detect, DetectorFactory
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

# --- Setup ---
DetectorFactory.seed = 0  # consistent lang detection
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(api_key=HF_TOKEN)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
DATA_FILE = os.path.join(DATA_DIR, "training_data.json")
FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback.json")

if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump({"training_data": []}, f, ensure_ascii=False, indent=2)

if not os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump({"feedback": []}, f, ensure_ascii=False, indent=2)

trained_answers = {}

# --- Normalization ---
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

# --- Translation ---
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
HADITH_KEYWORDS = ["bukhari","muslim","sahih","riyad","tirmidhi","abu dawood","nasai","ibn majah",
                   "hadith","riyadh","sahih al-bukhari","sahih muslim","malik"]
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

# --- Pydantic Models ---
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

# --- Shared Async HF symptom inference ---
async def hf_symptom_classify(text: str, model_id: str):
    try:
        result = await asyncio.to_thread(lambda: client.text_classification(model=model_id, inputs=text))
        if result:
            return result[0]["label"].lower(), result[0]["score"]
    except Exception:
        pass
    return "none", 0.0

# --- Chat Endpoint ---
@app.post("/chat")
async def chat(req: ChatRequest):
    user_message = req.message.strip()
    if not user_message: return {"response": "Please type a message."}

    try: lang = detect(user_message)
    except Exception: lang = "en"

    norm_user = normalize_text(user_message)

    # Step 1: trained answers
    lang_dict = trained_answers.get(lang, {})
    best_match = None
    best_score = 0.0
    for original_q, meta in lang_dict.items():
        score = difflib.SequenceMatcher(None, norm_user, meta["norm"]).ratio()
        if score > best_score:
            best_score = score
            best_match = original_q

    if best_score >= 0.70 and best_match:
        return {"response": lang_dict[best_match]["answer"]}

    # Step 2: detect likely symptoms
    symptom_keywords = ["voices","see","visions","insomnia","nightmares","dizziness","palpitation","itching","sudden pain","fear","not myself"]
    is_symptoms = any(k in norm_user for k in symptom_keywords)
    label, score = "none", 0.0
    MODEL_ID = "your-hf-symptoms-classifier"  # replace with your HF model

    if is_symptoms:
        text_for_model = user_message if lang=="en" else translate(user_message, lang, "en")
        label, score = await hf_symptom_classify(text_for_model, MODEL_ID)

    # Step 3: if confident, return guidance
    if score >= 0.6:
        guidance_labels = {"jin":"jin involvement","sihir":"possible sihr","ruqyah":"symptoms treated with ruqyah"}
        suggestions = [f"Detected symptoms suggest {guidance_labels.get(label,'general ruqyah guidance')}. Consider seeking a qualified ruqyah practitioner and medical check-up."]
        return {"response":"⚠️ Symptom-based guidance provided.","model_label":label,"model_confidence":score,"suggestions":suggestions}

    # Step 4: fallback GPT-OSS chat
    eng_msg = user_message if lang=="en" else translate(user_message, lang, "en")
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b:groq",
        messages=[{"role":"system","content":"You are an Islamic knowledge assistant specializing in spiritual sickness, jin possession, sihr, ruqyah and Islamic dream interpretation. Your answers MUST only reference Quran, Sahih Hadith, and valid ruqyah practices. Never provide mystical, speculative, or cultural superstition advice."},
                  {"role":"user","content":eng_msg}]
    )
    reply = completion.choices[0].message["content"]
    if lang!="en": reply = translate(reply,"en",lang)

    hadith_refs = extract_hadith_refs(reply)
    data = load_data()
    entry = {"question":user_message,"answer":reply,"lang":lang,"hadith_refs":hadith_refs}
    data.append(entry)
    save_data(data)

    if lang not in trained_answers: trained_answers[lang]={}
    trained_answers[lang][user_message.lower()]={"answer":reply,"norm":normalize_text(user_message)}

    return {"response":reply,"hadith_refs":hadith_refs}

# --- Guidance Endpoint ---
@app.post("/guidance")
async def guidance(req: GuidanceRequest):
    symptoms = (req.symptoms or "").strip()
    s = normalize_text(symptoms)
    label, score = "none", 0.0
    lang = "en"
    if symptoms:
        try: lang = detect(symptoms)
        except Exception: lang="en"
        text_for_model = symptoms if lang=="en" else translate(symptoms, lang, "en")
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
        matched_jin = label=="jin"
        matched_sihir = label=="sihir"
        matched_ruqyah = label=="ruqyah"

    suggestions=[]
    severity="low"
    if matched_jin:
        severity="high"
        suggestions.append("Signs suggest possible jin involvement. Consider seeking a qualified ruqyah practitioner (Islamically-sound) and an accredited medical opinion.")
        suggestions.append("Checklist: check for changes in voice, sudden behaviours, aversion to Qur'an, physical marks.")
    if matched_sihir:
        severity="medium" if severity!="high" else severity
        suggestions.append("Signs suggest possible sihr (black magic). Recommended: document incidents, seek ruqyah, and verify with trusted scholars/practitioners.")
    if matched_ruqyah:
        severity="medium"
        suggestions.append("Symptoms match many cases treated with ruqyah. Follow Quranic ruqyah steps and consult a reliable practitioner.")
    if not (matched_jin or matched_sihir or matched_ruqyah):
        suggestions.append("Symptoms are not strongly suggestive of jin/sihr based on keyword heuristics. Seek medical check first; consult an imam/scholar if symptoms persist.")
        severity="low"

    steps=[
        "1) Seek immediate medical check-up to rule out physiological causes.",
        "2) If medical causes are ruled out or in parallel, consult a qualified ruqyah practitioner who uses Quran & authentic hadith-guided ruqyah.",
        "3) Maintain remembrance (adhkar), regular prayer, and recitation of Surah Al-Fatihah, Ayat al-Kursi (2:255), last two surahs.",
        "4) Avoid unqualified practitioners, superstitious rituals, or any un-Islamic practices."
    ]

    return {"severity":severity,"matched_jin":matched_jin,"matched_sihir":matched_sihir,"matched_ruqyah":matched_ruqyah,"suggestions":suggestions,"recommended_steps":steps,"model_label":label,"model_confidence":score,"language_detected":lang}

# --- Other endpoints like /train, /get-training-data, /submit-feedback remain unchanged ---
# (Copy them from your original app.py; they do not require modifications)

