from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
from typing import List
import os

DB_FILE = "chatbot_data.db"

# Ensure DB exists
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Feedback table
    c.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        student_id TEXT,
        program TEXT,
        q4 TEXT,q5 TEXT,q6 TEXT,q7 TEXT,q8 TEXT,q9 TEXT,q10 TEXT,q11 TEXT,
        q12 TEXT,q13 TEXT,q14 TEXT
    )""")
    # Training table
    c.execute("""
    CREATE TABLE IF NOT EXISTS training_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT UNIQUE,
        answer TEXT,
        lang TEXT
    )""")
    conn.commit()
    conn.close()

init_db()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class Feedback(BaseModel):
    name: str = None
    student_id: str = None
    program: str = None
    q4: str = None; q5: str = None; q6: str = None; q7: str = None; q8: str = None; q9: str = None; q10: str = None; q11: str = None
    q12: str = None; q13: str = None; q14: str = None

class TrainData(BaseModel):
    question: str
    answer: str
    lang: str = "en"

class ChatMessage(BaseModel):
    message: str

# Chatbot placeholder
@app.post("/chat")
async def chat(msg: ChatMessage):
    # For now, echo
    return {"response": f"Bot response to: {msg.message}"}

# Submit feedback
@app.post("/submit-feedback")
async def submit_feedback(feedback: Feedback):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    fields = [feedback.name, feedback.student_id, feedback.program,
              feedback.q4, feedback.q5, feedback.q6, feedback.q7, feedback.q8, feedback.q9, feedback.q10, feedback.q11,
              feedback.q12, feedback.q13, feedback.q14]
    c.execute("""
        INSERT INTO feedback (name,student_id,program,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q14)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, fields)
    conn.commit()
    conn.close()
    return {"status":"ok"}

# Get all feedback
@app.get("/get-feedback")
async def get_feedback():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM feedback ORDER BY id DESC")
    rows = c.fetchall()
    keys = [col[0] for col in c.description]
    feedback = [dict(zip(keys,row)) for row in rows]
    conn.close()
    return feedback

# Add training data
@app.post("/train")
async def train_data(data: TrainData):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute("INSERT OR REPLACE INTO training_data (question,answer,lang) VALUES (?,?,?)",
                  (data.question,data.answer,data.lang))
        conn.commit()
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=str(e))
    conn.close()
    return {"status":"ok"}

# Get training data
@app.get("/get-training-data")
async def get_training_data():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM training_data ORDER BY id DESC")
    rows = c.fetchall()
    keys = [col[0] for col in c.description]
    training_data = [dict(zip(keys,row)) for row in rows]
    conn.close()
    return {"training_data": training_data}
