from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langdetect import detect
import re

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Simple dataset for now ===
knowledge_base = {
    "en": {
        "sihir": "According to Islamic teachings, sihir (black magic) is real and can affect people. Recite Surah Al-Baqarah, Ayat Kursi, and perform ruqyah regularly.",
        "jinn": "Jinn can cause disturbances, but Allah gives protection through dhikr, dua, and reciting Surah Al-Falaq and An-Naas.",
        "ruqyah": "Ruqyah should be done using Quranic verses and authentic duas — avoid any form of shirk or amulets.",
        "spiritual": "Spiritual sickness refers to diseases of the heart, such as envy, pride, or doubt — purified through repentance and remembrance of Allah."
    },
    "ms": {
        "sihir": "Menurut ajaran Islam, sihir itu wujud dan boleh mempengaruhi manusia. Bacalah Surah Al-Baqarah, Ayat Kursi, dan lakukan ruqyah secara berkala.",
        "jin": "Jin boleh mengganggu manusia, tetapi Allah memberi perlindungan melalui zikir, doa, serta bacaan Surah Al-Falaq dan An-Naas.",
        "ruqyah": "Ruqyah dilakukan dengan ayat-ayat Al-Quran dan doa sahih — elakkan tangkal atau jampi yang syirik.",
        "spiritual": "Penyakit rohani ialah penyakit hati seperti hasad, takabbur, dan ragu-ragu — dirawat dengan taubat dan zikir kepada Allah."
    },
    "ar": {
        "سحر": "السحر حقيقي ويمكن أن يؤثر على الناس. الرقية الشرعية وقراءة سورة البقرة وآية الكرسي تحمي المسلم من الأذى.",
        "جن": "الجن قد يؤذون الإنسان، لكن الله يحمي من خلال الذكر والدعاء وقراءة سور الفلق والناس.",
        "رقية": "يجب أن تكون الرقية بالقرآن والأدعية الصحيحة، وتجنب الشرك والتمائم.",
        "روح": "المرض الروحي يتعلق بأمراض القلب كالحسد والكبر والشك، ويعالج بالتوبة والذكر."
    }
}


class Message(BaseModel):
    text: str


@app.post("/chat")
async def chat(message: Message):
    text = message.text.strip()
    try:
        lang = detect(text)
    except:
        lang = "en"

    if lang.startswith("ms"):
        lang = "ms"
    elif lang.startswith("ar"):
        lang = "ar"
    else:
        lang = "en"

    response = None
    for keyword, reply in knowledge_base.get(lang, {}).items():
        if re.search(keyword, text, re.IGNORECASE):
            response = reply
            break

    if not response:
        if lang == "ms":
            response = "Maaf, saya hanya dapat menjawab berdasarkan rawatan Islam, ruqyah, sihir, dan gangguan jin."
        elif lang == "ar":
            response = "عذرًا، أستطيع الإجابة فقط بناءً على الرقية والسحر والجن والأمراض الروحية."
        else:
            response = "Sorry, I can only answer based on Islamic healing, ruqyah, sihir, and jinn-related issues."

    return {"response": response}


@app.post("/train")
async def train(request: Request):
    data = await request.json()
    question = data.get("question")
    answer = data.get("answer")
    lang = data.get("lang", "en")

    if lang not in knowledge_base:
        knowledge_base[lang] = {}

    knowledge_base[lang][question.lower()] = answer
    return {"message": f"Training added successfully for {lang}!"}
