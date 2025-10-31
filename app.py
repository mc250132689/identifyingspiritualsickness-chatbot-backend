from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
import os
from huggingface_hub import InferenceClient

app = FastAPI()

# Allow frontend (local or deployed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to SQLite DB
os.makedirs("data", exist_ok=True)
DB_PATH = "data/examples.db"
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute("CREATE TABLE IF NOT EXISTS examples (id INTEGER PRIMARY KEY, question TEXT, answer TEXT)")
conn.commit()
conn.close()

# Model for chat messages
class Message(BaseModel):
    message: str

class TrainData(BaseModel):
    question: str
    answer: str

# Setup HuggingFace inference
client = InferenceClient(provider="groq", api_key=os.getenv("HF_TOKEN"))

@app.get("/")
def root():
    return {"message": "Spiritual Sickness Chatbot Backend is running."}

@app.post("/chat")
def chat(msg: Message):
    user_text = msg.message

    # First, try matching from DB
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT answer FROM examples WHERE question LIKE ?", ('%' + user_text + '%',))
    result = c.fetchone()
    conn.close()

    if result:
        return {"reply": result[0]}

    # If no match, call AI model
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": user_text}],
    )

    reply = completion.choices[0].message["content"]
    return {"reply": reply}

@app.post("/chat/train")
def train(data: TrainData):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO examples (question, answer) VALUES (?, ?)", (data.question, data.answer))
    conn.commit()
    conn.close()
    return {"message": "Training data added successfully."}
