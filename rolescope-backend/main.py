"""
RoleScope AI — Backend API

FastAPI service powering RoleScope AI: analyzes uploaded resumes against
job descriptions, generates ATS-style feedback, and matches users to
career roles using a deterministic scoring model.
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
import math
import google.generativeai as genai
import pypdf
from docx import Document
import json
import io
import os

# --------------------------------------------------------------------------
# App configuration
# --------------------------------------------------------------------------

app = FastAPI(
    title="RoleScope AI API",
    description="AI-powered resume analysis and career role matching service.",
    version="1.0.0",
)

# Allow the frontend (deployed separately) to call this API.
# Restrict allow_origins to the production frontend domain before scaling.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini API key is injected via environment variable (set on the hosting platform).
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

GEMINI_MODEL = "models/gemini-1.5-flash"


# --------------------------------------------------------------------------
# File parsing utilities
# --------------------------------------------------------------------------

async def read_file(file: UploadFile) -> str:
    """Extract raw text from an uploaded PDF or DOCX resume."""
    content = await file.read()

    if file.filename.endswith(".pdf"):
        pdf_reader = pypdf.PdfReader(io.BytesIO(content))
        return "".join(page.extract_text() or "" for page in pdf_reader.pages)

    if file.filename.endswith(".docx"):
        doc = Document(io.BytesIO(content))
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)

    return ""


def get_gemini_json(prompt: str, content_parts: List[str]) -> dict:
    """Send a prompt to Gemini and parse the response as JSON."""
    try:
        model = genai.GenerativeModel(
            GEMINI_MODEL,
            generation_config={"response_mime_type": "application/json"},
        )
        response = model.generate_content([prompt] + content_parts)
        text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        return {"error": str(e)}


# --------------------------------------------------------------------------
# Career Compass — deterministic role-matching model
# --------------------------------------------------------------------------

class CompassPayload(BaseModel):
    """User responses to the Career Compass questionnaire."""
    answers: Dict[str, Dict[str, int]]


# Reference vectors describing each role across six behavioral dimensions.
# Used to compute similarity against the user's own profile.
ROLE_PROFILES = {
    "Product Manager": {"ambiguity": 4, "people": 3, "depth": 3, "pace": 2, "autonomy": 3, "motivation": 4},
    "Data Analyst": {"ambiguity": 2, "people": 1, "depth": 5, "pace": 2, "autonomy": 3, "motivation": 3},
    "Growth Marketer": {"ambiguity": 4, "people": 3, "depth": 2, "pace": 4, "autonomy": 3, "motivation": 4},
    "Software Engineer": {"ambiguity": 2, "people": 1, "depth": 5, "pace": 3, "autonomy": 4, "motivation": 3},
    "UX Designer": {"ambiguity": 3, "people": 4, "depth": 3, "pace": 3, "autonomy": 3, "motivation": 4},
    "Sales Specialist": {"ambiguity": 3, "people": 5, "depth": 1, "pace": 5, "autonomy": 2, "motivation": 5},
    "Project Manager": {"ambiguity": 2, "people": 5, "depth": 2, "pace": 4, "autonomy": 2, "motivation": 4},
}


def calculate_similarity(user_vector: dict, role_vector: dict) -> float:
    """Compute cosine similarity between a user's profile and a role profile."""
    keys = set(user_vector.keys()) & set(role_vector.keys())
    dot_product = sum(user_vector[k] * role_vector[k] for k in keys)
    magnitude_user = math.sqrt(sum(v ** 2 for v in user_vector.values()))
    magnitude_role = math.sqrt(sum(v ** 2 for v in role_vector.values()))

    if magnitude_user == 0 or magnitude_role == 0:
        return 0

    return dot_product / (magnitude_user * magnitude_role)


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.get("/")
def health_check():
    """Basic health check endpoint."""
    return {"message": "RoleScope AI API is running"}


@app.post("/analiz-et")
async def analyze_resume(
    cv: UploadFile = File(...),
    ilan: str = Form(None),
):
    """
    Analyze an uploaded resume, optionally against a target job posting.

    Returns an ATS-style fit score, rejection risk factors, negotiation
    guidance, a suggested proof-of-skill project, matching role
    recommendations, and a ready-to-send outreach message.
    """
    try:
        cv_text = await read_file(cv)
        job_context = f"JOB POSTING: {ilan}" if ilan else "JOB POSTING: None (general analysis)"

        prompt = """
        ROLE: You are a career strategy advisor.
        TASK: Analyze the candidate's resume.
        OUTPUT FORMAT (JSON):
        {
            "skor": 75,
            "ats_red_sebebi": "Reason for potential ATS rejection...",
            "pazarlik_stratejisi": { "masadaki_kozlarin": "...", "dikkat_etmen_gerekenler": "...", "taktik_cumlesi": "..." },
            "teknik_kanit_gorevi": { "proje_fikri": "...", "nasil_yapilir": "...", "etkisi": "..." },
            "onerilen_isler": [ {"pozisyon": "...", "neden": "...", "arama_kodu": "..."} ],
            "linkedin_dm_mesaji": "...",
            "kategorize_edilmis_eksikler": [ {"eksik_yetenek": "X", "onem_derecesi": "Kritik", "cozum_yolu": "..."} ]
        }
        """

        return get_gemini_json(prompt, [f"RESUME: {cv_text}", job_context])

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/kariyer-pusulasi-eski")
async def career_compass_llm(
    bolum: str = Form(...),
    mod: str = Form(...),
    dna_skorlari: str = Form(...),
):
    """
    Legacy LLM-based role suggestion endpoint (superseded by the
    deterministic /career-compass endpoint below). Retained for reference.
    """
    try:
        dna_json = json.loads(dna_skorlari)

        if mod == "UZMANLAS":
            prompt = (
                f"ROLE: Career specialist. USER wants to go deeper in the '{bolum}' field. "
                f"PROFILE: {dna_json}. TASK: Suggest 3 related sub-roles. "
                "OUTPUT (JSON): [{'rol':'...','uyum':'...','neden':'...','arama_kodu':'...'}]"
            )
        else:
            prompt = (
                f"ROLE: Career coach. USER currently works in '{bolum}' but wants a change. "
                f"PROFILE: {dna_json}. TASK: Suggest 3 alternative careers. "
                "OUTPUT (JSON): [{'rol':'...','uyum':'...','neden':'...','arama_kodu':'...'}]"
            )

        return get_gemini_json(prompt, [])

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/career-compass")
def career_compass_analysis(payload: CompassPayload):
    """
    Deterministic career-matching endpoint (no LLM call).

    Aggregates the user's questionnaire answers into a six-dimensional
    profile vector, computes cosine similarity against each reference
    role profile, and returns the top three matches.
    """
    user_scores = {
        "ambiguity": 0, "people": 0, "depth": 0,
        "pace": 0, "autonomy": 0, "motivation": 0,
    }

    for scores in payload.answers.values():
        for category, score in scores.items():
            if category in user_scores:
                user_scores[category] += score

    matches = []
    for role, profile in ROLE_PROFILES.items():
        similarity = calculate_similarity(user_scores, profile)
        matches.append({
            "role": role,
            "match_score": round(similarity * 100),
            "details": profile,
        })

    matches.sort(key=lambda x: x["match_score"], reverse=True)

    return {
        "user_profile": user_scores,
        "top_matches": matches[:3],
    }
