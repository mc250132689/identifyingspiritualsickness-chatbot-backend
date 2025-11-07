from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from langdetect import detect
import difflib
import json
import os
import requests
from collections import Counter

# --- FastAPI setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Hugging Face GPT-OSS client ---
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(provider="groq", api_key=HF_TOKEN)

# --- Training data file ---
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
        lang = item.get("lang", "en")
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

# --- Request models ---
class ChatRequest(BaseModel):
    message: str

class TrainRequest(BaseModel):
    question: str
    answer: str
    lang: str = None

# --- Active WebSocket connections ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

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

    lang_dict = trained_answers.get(lang, {})
    match = difflib.get_close_matches(user_message.lower(), lang_dict.keys(), n=1, cutoff=0.6)
    if match:
        answer = lang_dict[match[0]]
    else:
        eng_msg = user_message if lang == "en" else translate(user_message, lang, "en")
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an Islamic assistant specializing in spiritual sickness, "
                        "sihr, jinn disturbances, dreams interpretation, and ruqyah guidance. "
                        "Always answer according to Quran, Sunnah, and authentic teachings."
                    ),
                },
                {"role": "user", "content": eng_msg},
            ],
        )
        answer = completion.choices[0].message["content"]
        if lang != "en":
            answer = translate(answer, "en", lang)

        data = load_data()
        data.append({"question": user_message, "answer": answer, "lang": lang})
        save_data(data)
        if lang not in trained_answers:
            trained_answers[lang] = {}
        trained_answers[lang][user_message.lower()] = answer

        # Broadcast new chat to WebSocket
        import asyncio
        asyncio.create_task(manager.broadcast({"type": "chat", "question": user_message, "answer": answer}))

    return {"response": answer}

# --- Train endpoint ---
@app.post("/train")
async def train(req: TrainRequest):
    question = req.question.strip()
    answer = req.answer.strip()
    lang = req.lang or "en"
    if not question or not answer:
        return {"message": "Please provide both question and answer."}
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
    if lang not in trained_answers:
        trained_answers[lang] = {}
    trained_answers[lang][question.lower()] = answer

    import asyncio
    asyncio.create_task(manager.broadcast({"type": "update", "question": question, "answer": answer, "lang": lang}))

    return {"message": f"{'Updated' if updated else 'Added'} {lang.upper()} training data successfully."}

# --- Delete training entry ---
@app.delete("/train")
async def delete_training(question: str = Query(...), lang: str = "en"):
    data = load_data()
    original_len = len(data)
    data = [item for item in data if not (item["lang"] == lang and item["question"].lower() == question.lower())]
    save_data(data)
    trained_answers.get(lang, {}).pop(question.lower(), None)

    import asyncio
    asyncio.create_task(manager.broadcast({"type": "delete", "question": question, "lang": lang}))

    return {"deleted": original_len - len(data)}

# --- Get all training data ---
@app.get("/get-training-data")
async def get_training_data():
    return {"training_data": load_data()}

# --- Admin analytics ---
ADMIN_KEY = os.getenv("ADMIN_KEY", "mc250132689")
@app.get("/admin-stats")
async def admin_stats(key: str = Query(...)):
    if key != ADMIN_KEY:
        return {"error": "Unauthorized"}
    data = load_data()
    total_records = len(data)
    lang_count = Counter(item.get("lang", "unknown") for item in data)
    avg_q_len = round(sum(len(item["question"]) for item in data) / total_records, 1) if total_records else 0
    avg_a_len = round(sum(len(item["answer"]) for item in data) / total_records, 1) if total_records else 0
    return {
        "total_records": total_records,
        "languages": dict(lang_count),
        "avg_question_length": avg_q_len,
        "avg_answer_length": avg_a_len
    }

# --- WebSocket for live admin ---
@app.websocket("/ws/admin")
async def websocket_admin(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep connection open
    except WebSocketDisconnect:
        manager.disconnect(websocket)
