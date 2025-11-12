from fastapi import FastAPI, Query, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
from typing import List, Optional

DB_FILE = "chatbot_data.db"

# --- Initialize SQLite DB ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Training data table
    c.execute('''CREATE TABLE IF NOT EXISTS training_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        lang TEXT DEFAULT 'en'
    )''')
    # Feedback table
    c.execute('''CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        student_id TEXT,
        program TEXT,
        q4 TEXT,
        q5 TEXT,
        q6 TEXT,
        q7 TEXT,
        q8 TEXT,
        q9 TEXT,
        q10 TEXT,
        q11 TEXT,
        q12 TEXT,
        q13 TEXT,
        q14 TEXT
    )''')
    conn.commit()
    conn.close()

init_db()

# --- FastAPI setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic models ---
class TrainData(BaseModel):
    question: str
    answer: str
    lang: Optional[str] = 'en'

class FeedbackData(BaseModel):
    name: Optional[str] = ""
    student_id: Optional[str] = ""
    program: Optional[str] = ""
    q4: Optional[str] = ""
    q5: Optional[str] = ""
    q6: Optional[str] = ""
    q7: Optional[str] = ""
    q8: Optional[str] = ""
    q9: Optional[str] = ""
    q10: Optional[str] = ""
    q11: Optional[str] = ""
    q12: Optional[str] = ""
    q13: Optional[str] = ""
    q14: Optional[str] = ""

# --- Helper ---
def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

# --- Endpoints ---
@app.post("/train")
async def train_data(data: TrainData):
    conn = get_db_connection()
    c = conn.cursor()
    existing = c.execute("SELECT id FROM training_data WHERE LOWER(question)=?", (data.question.lower(),)).fetchone()
    if existing:
        c.execute("UPDATE training_data SET answer=?, lang=? WHERE id=?", (data.answer, data.lang, existing["id"]))
    else:
        c.execute("INSERT INTO training_data (question, answer, lang) VALUES (?, ?, ?)", (data.question, data.answer, data.lang))
    conn.commit()
    conn.close()
    return {"message": "Training data saved successfully."}

@app.get("/get-training-data")
async def get_training_data():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT question, answer, lang FROM training_data ORDER BY id ASC")
    data = [dict(row) for row in c.fetchall()]
    conn.close()
    return {"training_data": data}

@app.post("/submit-feedback")
async def submit_feedback(fb: FeedbackData):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        INSERT INTO feedback (name, student_id, program, q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q14)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    ''', (fb.name, fb.student_id, fb.program, fb.q4,fb.q5,fb.q6,fb.q7,fb.q8,fb.q9,fb.q10,fb.q11,fb.q12,fb.q13,fb.q14))
    conn.commit()
    conn.close()
    return {"message": "Feedback submitted successfully."}

@app.get("/get-feedback")
async def get_feedback():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM feedback ORDER BY id ASC")
    data = [dict(row) for row in c.fetchall()]
    conn.close()
    return data

@app.get("/guidance")
async def guidance(question: str = Query(...)):
    return {"response": f"Guidance response for: {question}"}

@app.get("/ping")
async def ping():
    return {"status": "alive"}
