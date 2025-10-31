from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langdetect import detect
import json
import os

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_FILE = "training_data.json"

# Initialize training data file if not exists
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)


class ChatRequest(BaseModel):
    message: str


class TrainRequest(BaseModel):
    question: str
    answer: str


def load_data():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@app.post("/chat")
async def chat(req: ChatRequest):
    user_message = req.message.strip()
    lang = detect(user_message)
    data = load_data()

    # Try matching by language
    for item in data:
        if item["lang"] == lang and item["question"].lower() in user_message.lower():
            return {"response": item["answer"]}

    # Default fallback response
    default_responses = {
        "en": "I’m sorry, I don’t yet have an Islamic answer for that. Please refer to the Quran, Sunnah, or ruqyah healing practices.",
        "ms": "Maaf, saya belum mempunyai jawapan Islam untuk itu. Sila rujuk kepada al-Quran, Sunnah, atau rawatan ruqyah.",
        "ar": "عذرًا، لا أملك إجابة إسلامية لذلك بعد. يرجى الرجوع إلى القرآن والسنة والرقية الشرعية.",
    }

    return {"response": default_responses.get(lang, default_responses["en"])}


@app.post("/train")
async def train(req: TrainRequest):
    question = req.question.strip()
    answer = req.answer.strip()
    lang = detect(question)

    if not question or not answer:
        return {"message": "Please provide both question and answer."}

    data = load_data()

    # Update if similar question already exists in same language
    for item in data:
        if item["lang"] == lang and item["question"].lower() == question.lower():
            item["answer"] = answer
            save_data(data)
            return {"message": f"Updated existing {lang.upper()} training data."}

    data.append({"question": question, "answer": answer, "lang": lang})
    save_data(data)
    return {"message": f"Added new {lang.upper()} training data successfully."}
