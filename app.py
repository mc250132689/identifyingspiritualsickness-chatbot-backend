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
import aiofiles
import subprocess
import datetime
import aiohttp

app = FastAPI()

# ======= Allow frontend origins =======
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======= DATA + BACKUP SETTINGS =======
DATA_DIR = "data"
GITHUB_REPO = "https://github.com/mc250132689/identifyingspiritualsickness-chatbot-backend"
GITHUB_BRANCH = "main"
BACKUP_INTERVAL = 180  # 3 min
PING_INTERVAL = 180  # 3 min

os.makedirs(DATA_DIR, exist_ok=True)

# ======= JSON I/O Helpers =======
async def read_json(file):
    path = os.path.join(DATA_DIR, file)
    if not os.path.exists(path):
        async with aiofiles.open(path, "w") as f:
            await f.write("[]")
    async with aiofiles.open(path, "r") as f:
        content = await f.read()
        return json.loads(content or "[]")

async def write_json(file, data):
    path = os.path.join(DATA_DIR, file)
    async with aiofiles.open(path, "w") as f:
        await f.write(json.dumps(data, indent=2))

# ======= Example: Chatbot Endpoint =======
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    user_message = req.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Empty message")

    # Simulate model output
    response = f"🤖 The chatbot received: {user_message}"
    return {"response": response}

# ======= Example: Feedback submission =======
class Feedback(BaseModel):
    name: str
    student_id: str
    feedback: str

@app.post("/submit-feedback")
async def submit_feedback(data: Feedback):
    feedback_data = await read_json("feedback.json")
    new_entry = {
        "name": data.name,
        "student_id": data.student_id,
        "feedback": data.feedback,
    }
    feedback_data.append(new_entry)
    await write_json("feedback.json", feedback_data)
    return {"status": "success", "message": "Feedback submitted successfully"}

@app.get("/get-feedback")
async def get_feedback():
    feedback_data = await read_json("feedback.json")
    return {"feedback": feedback_data}

# ======= Example: Training data handling =======
class TrainingItem(BaseModel):
    question: str
    answer: str

@app.post("/train")
async def add_training(data: TrainingItem):
    training_data = await read_json("training_data.json")
    new_item = {"question": data.question, "answer": data.answer}
    training_data.append(new_item)
    await write_json("training_data.json", training_data)
    return {"status": "success", "message": "Training data added successfully"}

@app.get("/get-training")
async def get_training():
    training_data = await read_json("training_data.json")
    return {"training": training_data}

# ======= Background: Keep Server Awake =======
async def keep_awake():
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://identifyingspiritualsickness-chatbot.onrender.com/") as r:
                    print(f"[KEEP_ALIVE] Pinged server: {r.status}")
        except Exception as e:
            print("[KEEP_ALIVE] Error:", e)
        await asyncio.sleep(PING_INTERVAL)

# ======= Background: Auto Backup JSONs =======
async def auto_backup():
    while True:
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[BACKUP] Starting backup at {timestamp}")

            subprocess.run(["git", "config", "--global", "user.email", "bot@example.com"])
            subprocess.run(["git", "config", "--global", "user.name", "BackupBot"])

            if not os.path.exists(".git"):
                subprocess.run(["git", "init"])
                subprocess.run(["git", "remote", "add", "origin", GITHUB_REPO])

            subprocess.run(["git", "add", DATA_DIR])
            subprocess.run(["git", "commit", "-m", f"Auto backup {timestamp}"], check=False)
            subprocess.run(["git", "push", "origin", GITHUB_BRANCH, "--force"], check=False)

            print("[BACKUP] Backup completed and pushed to GitHub.")
        except Exception as e:
            print("[BACKUP ERROR]", e)
        await asyncio.sleep(BACKUP_INTERVAL)

# ======= Background: Auto Fetch GitHub Backup =======
async def auto_fetch():
    try:
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR, exist_ok=True)
        if not os.path.exists(".git"):
            subprocess.run(["git", "clone", GITHUB_REPO, "."], check=False)
        else:
            subprocess.run(["git", "pull", "origin", GITHUB_BRANCH], check=False)
        print("[FETCH] Synced latest backup from GitHub.")
    except Exception as e:
        print("[FETCH ERROR]", e)

# ======= Startup & Shutdown Events =======
@app.on_event("startup")
async def startup_event():
    await auto_fetch()
    asyncio.create_task(keep_awake())
    asyncio.create_task(auto_backup())
    print("[STARTUP] Server started — keep-alive and backup running.")

@app.on_event("shutdown")
async def shutdown_event():
    print("[SHUTDOWN] Cleaning up.")
