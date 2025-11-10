from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import requests, os, base64, json, re

app = FastAPI()

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
    r = requests.get(GITHUB_API_URL, headers=headers).json()
    decoded_bytes = base64.b64decode(r["content"])
    return json.loads(decoded_bytes.decode("utf-8")), r["sha"]

def save_training_data(data, sha):
    encoded = base64.b64encode(json.dumps(data, indent=4).encode()).decode()
    payload = {"message": "Updated training data", "content": encoded, "sha": sha}
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}
    requests.put(GITHUB_API_URL, headers=headers, data=json.dumps(payload))

@app.get("/get-training-data")
def get_training():
    data, _ = load_training_data()
    return data

@app.post("/train")
async def train(request: Request):
    req = await request.json()
    q = req.get("question")
    a = req.get("answer")
    data, sha = load_training_data()
    data.append({"question": q, "answer": a})
    save_training_data(data, sha)
    return {"status": "success"}

@app.post("/chat")
async def chat(request: Request):
    req = await request.json()
    user_text = req.get("message", "").lower()

    data, _ = load_training_data()

    best_answer = None
    best_score = 0

    for item in data:
        question = item["question"].lower()
        words = user_text.split()
        score = sum(question.count(w) for w in words)
        if score > best_score:
            best_score = score
            best_answer = item["answer"]

    if best_answer:
        return {"reply": best_answer}

    return {"reply": "I do not have the exact answer yet. Please rephrase or provide more context."}

@app.post("/search-hadith")
async def search_hadith(request: Request):
    req = await request.json()
    query = req.get("query", "").lower()

    # Local hadith dataset (expand later)
    hadith_list = [
        {"text": "Actions are judged by intentions.", "source": "Sahih Bukhari"},
        {"text": "The cure for ignorance is to ask.", "source": "Sunan Abi Dawud"},
        {"text": "Indeed in the remembrance of Allah do hearts find rest.", "source": "Quran 13:28"}
    ]

    results = [h for h in hadith_list if query in h["text"].lower()]
    return {"results": results}

@app.get("/admin/get-data")
def admin_get():
    data, _ = load_training_data()
    return data

@app.post("/admin/delete")
async def admin_delete(request: Request):
    req = await request.json()
    index = req.get("index")

    data, sha = load_training_data()
    if 0 <= index < len(data):
        data.pop(index)
        save_training_data(data, sha)
        return {"status": "deleted"}

    return {"status": "error", "message": "Invalid index"}

@app.post("/feedback")
async def feedback(request: Request):
    req = await request.json()
    with open("/mnt/data/user_feedback.json", "a") as f:
        f.write(json.dumps(req) + "\n")
    return {"status": "received"}
