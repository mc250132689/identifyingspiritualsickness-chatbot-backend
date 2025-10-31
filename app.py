from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient
import os

# Initialize FastAPI
app = FastAPI(title="🕋 Spiritual Sickness Chatbot API")

# CORS (allow frontend connection)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to HuggingFace Inference API (Groq provider)
client = InferenceClient(
    provider="groq",
    api_key=os.environ.get("HF_TOKEN"),  # set this in Render environment
)

@app.get("/")
def home():
    return {
        "message": "🕋 Spiritual Sickness Chatbot API is running. Only responds to questions on rawatan Islam, sihir, jin, and ruqyah."
    }

@app.post("/chat")
async def chat(request: Request):
    """
    Chat endpoint — receives user message, filters it, and responds using GPT-OSS-20B
    """
    data = await request.json()
    user_message = data.get("message", "").strip()

    # Define Islamic spiritual sickness context keywords
    allowed_keywords = [
        "sihir", "jin", "ruqyah", "gangguan", "mimpi", "azab",
        "sakit", "rawatan", "doa", "quran", "zikir", "syifa", "roh", "hati"
    ]

    # If message not relevant, gently refuse
    if not any(k in user_message.lower() for k in allowed_keywords):
        return {
            "reply": (
                "⚠️ Maaf, saya hanya menjawab persoalan berkaitan dengan **rawatan Islam**, "
                "**sihir**, **gangguan jin**, dan **penyakit rohani**, berdasarkan al-Qur’an dan as-Sunnah. "
                "Sila tanya soalan dalam topik tersebut."
            )
        }

    # System instruction to keep model in Islamic context
    system_prompt = (
        "Anda ialah pembantu perubatan Islam yang hanya menjawab berdasarkan al-Qur'an, as-Sunnah, "
        "dan kaedah ruqyah syar'iyyah. "
        "Gunakan bahasa Melayu yang lembut, sopan, dan menenangkan. "
        "Jangan berikan nasihat moden, perubatan saintifik, atau kepercayaan khurafat. "
        "Fokus kepada doa, ayat ruqyah, tanda-tanda sihir, gangguan jin, dan cara rawatan Islam. "
        "Jika soalan di luar konteks, beritahu dengan sopan bahawa anda hanya menjawab dalam bidang rawatan Islam."
    )

    # Generate the completion from HuggingFace
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )

    reply = completion.choices[0].message["content"]
    return {"reply": reply}


@app.post("/train")
async def train(request: Request):
    """
    Dummy training endpoint — can be extended later
    """
    data = await request.json()
    question = data.get("question")
    answer = data.get("answer")
    # Here you could save to DB or fine-tune storage
    return {"status": "success", "message": f"Latihan disimpan: '{question[:40]}...' "}
