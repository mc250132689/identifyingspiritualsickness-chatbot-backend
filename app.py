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

# === Hugging Face GPT-OSS client (Groq) ===
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(api_key=HF_TOKEN)  # DO NOT set provider here

DATA_FILE = "training_data.json"
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump({"training_data": []}, f, ensure_ascii=False, indent=2)

trained_answers = {}

def load_data():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data_list = json.load(f)["training_data"]
    global trained_answers
    trained_answers = {}
    for item in data_list:
        lang = item["lang"]
        q = item["question"].lower()
        a = item["answer"]
        if lang not in trained_answers:
            trained_answers[lang] = {}
        trained_answers[lang][q] = a
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

    lang_dict = trained_answers.get(lang, {})
    match = difflib.get_close_matches(user_message.lower(), lang_dict.keys(), n=1, cutoff=0.6)
    if match:
        return {"response": lang_dict[match[0]]}

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

    if lang not in trained_answers:
        trained_answers[lang] = {}
    trained_answers[lang][user_message.lower()] = reply

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
        if item["lang"] == lang and item["question"].lower() == question.lower():
            item["answer"] = answer
            updated = True
            break
    if not updated:
        data.append({"question": question, "answer": answer, "lang": lang})
    save_data(data)
    trained_answers[lang][question.lower()] = answer

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
    lang_count = Counter(item["lang"] for item in data)

    avg_q = round(sum(len(i["question"]) for i in data) / total, 1) if total else 0
    avg_a = round(sum(len(i["answer"]) for i in data) / total, 1) if total else 0

    return {
        "total_records": total,
        "languages": dict(lang_count),
        "avg_question_length": avg_q,
        "avg_answer_length": avg_a
    }
