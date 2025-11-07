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

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Inference Client (Groq via HuggingFace Hub) ===
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(api_key=HF_TOKEN)

DATA_FILE = "training_data.json"
trained_answers = {}

def load_data():
    global trained_answers

    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump({"training_data": []}, f, ensure_ascii=False, indent=2)

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data_list = json.load(f)["training_data"]

    trained_answers = {}
    for item in data_list:
        q = item["question"].strip().lower()
        a = item["answer"].strip()
        # Assign language if missing
        lang = item.get("lang")
        if not lang:
            try:
                lang = detect(item["question"])
            except:
                lang = "en"
            item["lang"] = lang

        if lang not in trained_answers:
            trained_answers[lang] = {}
        trained_answers[lang][q] = a

    save_data(data_list)
    return data_list

def save_data(data_list):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump({"training_data": data_list}, f, ensure_ascii=False, indent=2)

load_data()

def translate(text, source, target):
    model_map = {
        ("en","ms"):"Helsinki-NLP/opus-mt-en-ms",
        ("ms","en"):"Helsinki-NLP/opus-mt-ms-en",
        ("en","ar"):"Helsinki-NLP/opus-mt-en-ar",
        ("ar","en"):"Helsinki-NLP/opus-mt-ar-en",
        ("ms","ar"):"Helsinki-NLP/opus-mt-ms-ar",
        ("ar","ms"):"Helsinki-NLP/opus-mt-ar-ms"
    }
    if (source, target) not in model_map:
        return text

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    try:
        r = requests.post(
            f"https://api-inference.huggingface.co/models/{model_map[(source,target)]}",
            headers=headers,
            json={"inputs": text},
            timeout=60
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
    user_msg = req.message.strip()
    if not user_msg:
        return {"response": "Please type a message."}

    try:
        lang = detect(user_msg)
    except:
        lang = "en"

    lang_dict = trained_answers.get(lang, {})
    match = difflib.get_close_matches(user_msg.lower(), lang_dict.keys(), n=1, cutoff=0.6)
    if match:
        return {"response": lang_dict[match[0]]}

    eng_msg = user_msg if lang == "en" else translate(user_msg, lang, "en")

    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b:groq",
        messages=[
            {"role": "system", "content": (
                "You are an Islamic assistant specializing in spiritual sickness, ruqyah, sihr, and jinn. "
                "Answer only according to Quran and Sahih Sunnah. Do not mention opinions without evidence."
            )},
            {"role": "user", "content": eng_msg},
        ],
    )

    reply = completion.choices[0].message["content"]
    if lang != "en":
        reply = translate(reply, "en", lang)

    data = load_data()
    data.append({"question": user_msg, "answer": reply, "lang": lang})
    save_data(data)
    trained_answers[lang][user_msg.lower()] = reply

    return {"response": reply}

@app.post("/train")
async def train(req: TrainRequest):
    q = req.question.strip()
    a = req.answer.strip()

    if not q or not a:
        return {"message": "Please provide both question and answer."}

    try:
        lang = detect(q)
    except:
        lang = "en"

    data = load_data()
    for item in data:
        if item["lang"] == lang and item["question"].lower() == q.lower():
            item["answer"] = a
            save_data(data)
            trained_answers[lang][q.lower()] = a
            return {"message": "Updated training data."}

    data.append({"question": q, "answer": a, "lang": lang})
    save_data(data)

    if lang not in trained_answers:
        trained_answers[lang] = {}
    trained_answers[lang][q.lower()] = a

    return {"message": "Added new training data."}

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
    lang_count = Counter(item["lang"] for item in data)

    avg_q = round(sum(len(i["question"]) for i in data)/total, 1) if total else 0
    avg_a = round(sum(len(i["answer"]) for i in data)/total, 1) if total else 0

    return {
        "total_records": total,
        "languages": dict(lang_count),
        "avg_question_length": avg_q,
        "avg_answer_length": avg_a
    }
