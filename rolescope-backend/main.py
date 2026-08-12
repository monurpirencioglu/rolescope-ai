from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel  # YENİ EKLENDİ
from typing import Dict, List  # YENİ EKLENDİ
import math  # YENİ EKLENDİ
import google.generativeai as genai
import pypdf
from docx import Document
import json
import io
import os

# --- AYARLAR ---
app = FastAPI(title="RoleScope AI API")

# Güvenlik Ayarı (Frontend sitemizin buraya erişmesi için izin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Anahtarı (Render.com'dan okuyacak)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

MODEL_ADI = "models/gemini-1.5-flash"


# --- YARDIMCI FONKSİYONLAR (ESKİLER) ---
async def read_file(file: UploadFile):
    content = await file.read()
    if file.filename.endswith(".pdf"):
        pdf_reader = pypdf.PdfReader(io.BytesIO(content))
        return "".join([page.extract_text() or "" for page in pdf_reader.pages])
    elif file.filename.endswith(".docx"):
        doc = Document(io.BytesIO(content))
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        return ""


def get_gemini_json(prompt, content_parts):
    try:
        model = genai.GenerativeModel(MODEL_ADI, generation_config={"response_mime_type": "application/json"})
        response = model.generate_content([prompt] + content_parts)
        text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        return {"error": str(e)}


# --- YENİ EKLENEN VERİLER VE FONKSİYONLAR (KARİYER PUSULASI) ---

# 1. Veri Modeli
class CompassPayload(BaseModel):
    answers: Dict[str, Dict[str, int]]


# 2. Rol Profilleri (Matematiksel Eşleşme İçin)
ROLE_PROFILES = {
    "Product Manager": {"ambiguity": 4, "people": 3, "depth": 3, "pace": 2, "autonomy": 3, "motivation": 4},
    "Data Analyst": {"ambiguity": 2, "people": 1, "depth": 5, "pace": 2, "autonomy": 3, "motivation": 3},
    "Growth Marketer": {"ambiguity": 4, "people": 3, "depth": 2, "pace": 4, "autonomy": 3, "motivation": 4},
    "Software Engineer": {"ambiguity": 2, "people": 1, "depth": 5, "pace": 3, "autonomy": 4, "motivation": 3},
    "UX Designer": {"ambiguity": 3, "people": 4, "depth": 3, "pace": 3, "autonomy": 3, "motivation": 4},
    "Sales Specialist": {"ambiguity": 3, "people": 5, "depth": 1, "pace": 5, "autonomy": 2, "motivation": 5},
    "Project Manager": {"ambiguity": 2, "people": 5, "depth": 2, "pace": 4, "autonomy": 2, "motivation": 4}
}


# 3. Cosine Similarity Hesaplama
def calculate_similarity(user_vector, role_vector):
    keys = set(user_vector.keys()) & set(role_vector.keys())
    dot_product = sum(user_vector[k] * role_vector[k] for k in keys)
    magnitude_user = math.sqrt(sum(v ** 2 for v in user_vector.values()))
    magnitude_role = math.sqrt(sum(v ** 2 for v in role_vector.values()))
    if magnitude_user == 0 or magnitude_role == 0:
        return 0
    return dot_product / (magnitude_user * magnitude_role)


# --- ENDPOINTLER (Kapılar) ---

@app.get("/")
def home():
    return {"message": "RoleScope AI API Calisiyor! 🚀"}


@app.post("/analiz-et")
async def analiz_et(
        cv: UploadFile = File(...),
        ilan: str = Form(None)
):
    try:
        cv_text = await read_file(cv)
        ilan_text = f"İLAN METNİ: {ilan}" if ilan else "İLAN METNİ: Yok (Genel Analiz)"

        prompt = """
        ROL: Sen Kariyer Menajerisiniz.
        GÖREV: CV'yi analiz et.
        ÇIKTI FORMATI (JSON):
        {
            "skor": 75,
            "ats_red_sebebi": "Red sebebi...",
            "pazarlik_stratejisi": { "masadaki_kozlarin": "...", "dikkat_etmen_gerekenler": "...", "taktik_cumlesi": "..." },
            "teknik_kanit_gorevi": { "proje_fikri": "...", "nasil_yapilir": "...", "etkisi": "..." },
            "onerilen_isler": [ {"pozisyon": "...", "neden": "...", "arama_kodu": "..."} ],
            "linkedin_dm_mesaji": "...",
            "kategorize_edilmis_eksikler": [ {"eksik_yetenek": "X", "onem_derecesi": "Kritik", "cozum_yolu": "..."} ]
        }
        """

        result = get_gemini_json(prompt, [f"CV: {cv_text}", ilan_text])
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- ESKİ KARİYER PUSULASI (Gemini Kullanan - Yedek Olarak Kalsın) ---
@app.post("/kariyer-pusulasi-eski")
async def kariyer_pusulasi_eski(
        bolum: str = Form(...),
        mod: str = Form(...),
        dna_skorlari: str = Form(...)
):
    try:
        dna_json = json.loads(dna_skorlari)

        if mod == "UZMANLAS":
            prompt = f"ROL: Kariyer Uzmanı. KULLANICI: '{bolum}' alanında derinleşmek istiyor. DNA: {dna_json}. GÖREV: 3 alt rol öner. ÇIKTI (JSON): [{{'rol':'...','uyum':'...','neden':'...','arama_kodu':'...'}}]"
        else:
            prompt = f"ROL: Kariyer Koçu. KULLANICI: '{bolum}' ama değiştirmek istiyor. DNA: {dna_json}. GÖREV: 3 farklı meslek öner. ÇIKTI (JSON): [{{'rol':'...','uyum':'...','neden':'...','arama_kodu':'...'}}]"

        result = get_gemini_json(prompt, [])
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- YENİ 12 SORULUK KARİYER PUSULASI (Matematiksel - API Kullanmaz) ---
@app.post("/career-compass")
def career_compass_analysis(payload: CompassPayload):
    # Kullanıcının temel puanlarını sıfırla
    user_scores = {
        "ambiguity": 0, "people": 0, "depth": 0,
        "pace": 0, "autonomy": 0, "motivation": 0
    }

    # Frontend'den gelen cevapları topla
    for scores in payload.answers.values():
        for category, score in scores.items():
            if category in user_scores:
                user_scores[category] += score

    # Rollerle eşleştir
    matches = []
    for role, profile in ROLE_PROFILES.items():
        similarity = calculate_similarity(user_scores, profile)
        match_score = round(similarity * 100)

        matches.append({
            "role": role,
            "match_score": match_score,
            "details": profile
        })

    # En yüksek puana göre sırala ve ilk 3'ü döndür
    matches.sort(key=lambda x: x["match_score"], reverse=True)

    return {
        "user_profile": user_scores,
        "top_matches": matches[:3]
    }