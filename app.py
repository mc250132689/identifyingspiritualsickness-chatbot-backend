from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import requests
import json
import os

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")
TRAINING_FILE_PATH = os.getenv("TRAINING_FILE_PATH")

GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{TRAINING_FILE_PATH}"

def load_training_data():
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}
    response = requests.get(GITHUB_API_URL, headers=headers).json()
    file_content = response["content"]
    decoded = json.loads((file_content.encode("ascii")))
    return json.loads(decoded)

def save_training_data(data):
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Content-Type": "application/json"
    }

    # Get current file SHA
    current = requests.get(GITHUB_API_URL, headers=headers).json()
    sha = current["sha"]

    encoded_content = json.dumps(data, indent=4)

    payload = {
        "message": "Update training data via chatbot learning",
        "content": encoded_content.encode("ascii"),
        "sha": sha
    }

    requests.put(GITHUB_API_URL, headers=headers, data=json.dumps(payload))

@app.get("/get-training-data")
def get_training():
    return load_training_data()

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_text = data.get("message", "").lower()

    training_data = load_training_data()

    # Simple semantic matching
    best_answer = None
    highest_score = 0

    for item in training_data:
        question = item["question"].lower()
        match_score = sum(1 for word in user_text.split() if word in question)
        if match_score > highest_score:
            highest_score = match_score
            best_answer = item["answer"]

    if best_answer:
        return {"reply": best_answer}
    else:
        return {"reply": "I do not have enough knowledge to answer that yet. Please provide more context or rephrase your question."}

@app.post("/train")
async def train(request: Request):
    data = await request.json()
    question = data.get("question")
    answer = data.get("answer")

    training_data = load_training_data()
    training_data.append({"question": question, "answer": answer})
    save_training_data(training_data)

    return {"status": "success", "message": "Training data added successfully"}
