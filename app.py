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

    # Language Detection
    def detect_lang(text):
        malay_words = ["apa", "bagaimana", "kenapa", "saya", "awak", "ruqyah", "jin", "sihir"]
        arabic_chars = re.compile("[\u0600-\u06FF]+")
        if arabic_chars.search(text):
            return "ar"
        if any(w in text for w in malay_words):
            return "ms"
        return "en"

    lang = detect_lang(user_text)

    # Spiritual Sickness Classification
    categories = {
        "hasad": ["jealous", "envy", "hasad", "pandangan", "mata", "dipandang"],
        "sihir": ["black magic", "magic", "santau", "guna", "guni-guni", "witchcraft", "sihir"],
        "jinn": ["jin", "nightmares", "sleep paralysis", "bisikan", "whisper", "angin", "ketakutan"],
        "stress": ["stress", "anxiety", "depress", "letih", "penat", "burnout"]
    }

    def classify(text):
        results = {k: sum(text.count(w) for w in words) for k, words in categories.items()}
        best = max(results, key=results.get)
        return best if results[best] > 0 else None

    result = classify(user_text)

    ruqyah_guidance = {
        "hasad": {
            "ms": "Ini mungkin tanda hasad. Amalkan Surah Al-Falaq, An-Naas, Ayat Kursi setiap pagi dan petang.",
            "en": "This may indicate evil eye (hasad). Recite Al-Falaq, An-Naas, and Ayat Al-Kursi daily.",
            "ar": "قد يكون هذا من الحسد. اقرأ المعوذتين وآية الكرسي صباحًا ومساءً."
        },
        "sihir": {
            "ms": "Ini mungkin ada unsur sihir. Lakukan ruqyah dengan membaca Surah Al-Baqarah ayat 1-5 dan Ayat Kursi.",
            "en": "This may indicate black magic. Perform ruqyah with Surah Al-Baqarah and Ayat Al-Kursi.",
            "ar": "قد يكون سحرًا. قم بالرقية بسورة البقرة وآية الكرسي."
        },
        "jinn": {
            "ms": "Ini mungkin gangguan jin. Banyakkan istighfar, zikir pagi & petang, dan dengar Surah Al-Baqarah.",
            "en": "This may indicate jinn disturbance. Increase dhikr and play Surah Al-Baqarah.",
            "ar": "قد يكون مسّ من الجن. أكثر من الذكر واستمع إلى سورة البقرة."
        },
        "stress": {
            "ms": "Ini lebih kepada tekanan emosi. Amalkan doa perlindungan dan rehat secukupnya.",
            "en": "This appears emotional stress. Engage in dhikr and ensure rest.",
            "ar": "يبدو أنه إرهاق نفسي. أكثر من الذكر وخذ قسطًا من الراحة."
        }
    }

    if result:
        reply = ruqyah_guidance[result][lang]
        return {"reply": reply}

    # If no classification matched, fallback to trained data
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

    fallback = {
        "ms": "Saya belum mempunyai jawapan lengkap. Sila tambah soalan ini dalam latihan.",
        "en": "I do not have enough knowledge yet. Please try rephrasing.",
        "ar": "لا توجد إجابة حالياً. حاول إعادة الصياغة."
    }

    return {"reply": fallback[lang]}

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
