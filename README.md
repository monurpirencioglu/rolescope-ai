# RoleScope AI

**AI-powered career assistant for CV strategy, ATS simulation, and job role navigation.**

RoleScope AI analyzes a candidate's resume — optionally against a target job posting — and returns an ATS-style fit score, rejection risk factors, negotiation guidance, and a personalized role-matching report. It combines an LLM-driven resume analysis engine with a deterministic, similarity-based career matching model.

🔗 **Live Demo:** [rolescope-ai.vercel.app](https://rolescope-ai.vercel.app/)

<img width="752" height="534" alt="image" src="https://github.com/user-attachments/assets/3ca74668-fa02-43dd-be0f-daeb55f0c0c8" />



<img width="1287" height="642" alt="image" src="https://github.com/user-attachments/assets/a3818cb8-759c-445c-b868-8a018a914e63" />

<!-- Replace with an actual screenshot or short GIF of the app in use -->

---

## What It Does

- **Resume Analysis:** Upload a CV (PDF/DOCX), optionally paste a target job description, and receive an AI-generated fit score, ATS rejection risk factors, and a tailored improvement plan.
- **Career Compass:** A 6-dimension questionnaire (ambiguity tolerance, people-orientation, depth of focus, pace, autonomy, motivation) maps the user to a profile vector, which is then matched against reference role profiles using cosine similarity — no LLM call required for this part, making it fast and fully deterministic.
- **Actionable Output:** Negotiation talking points, a suggested proof-of-skill project, and a ready-to-send outreach message tailored to the user's target role.

---

## Tech Stack

**Backend:** `Python` `FastAPI` `Google Gemini API` `pypdf` `python-docx`
**Frontend:** `Next.js` `TypeScript`
**Deployment:** Backend on Render, Frontend on Vercel
**Analytics:** Google Analytics 4 (GA4)

---

## Architecture

```
rolescope-ai/
├── backend/          FastAPI service — resume parsing, LLM analysis, matching engine
└── frontend/         Next.js application — user-facing interface
```

The backend exposes three main endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/analiz-et` | POST | Parses an uploaded resume (PDF/DOCX) and returns an AI-generated fit analysis against an optional job posting |
| `/career-compass` | POST | Deterministic role-matching based on cosine similarity between the user's profile and reference role vectors |
| `/kariyer-pusulasi-eski` | POST | Legacy LLM-based role suggestion endpoint, retained for reference |

---

## How the Matching Model Works

Each role (Product Manager, Data Analyst, Growth Marketer, etc.) is represented as a 6-dimensional reference vector across behavioral traits. A user's questionnaire responses are aggregated into their own vector, and **cosine similarity** is computed against every role profile to produce a ranked match score — the same class of technique used in recommendation systems, applied here to career fit rather than product recommendations.

```python
def calculate_similarity(user_vector, role_vector):
    dot_product = sum(user_vector[k] * role_vector[k] for k in keys)
    magnitude_user = math.sqrt(sum(v ** 2 for v in user_vector.values()))
    magnitude_role = math.sqrt(sum(v ** 2 for v in role_vector.values()))
    return dot_product / (magnitude_user * magnitude_role)
```

---

## Product Analytics & Iteration

After launch, I instrumented the app with GA4 and reviewed real usage data (450+ users). The analysis surfaced a **90%+ bounce rate**, indicating a gap between initial traffic and sustained engagement. Based on this finding, I'm currently implementing:

- **SEO improvements** (metadata, keyword targeting) to improve organic discovery
- **Onboarding UX changes** to increase first-session engagement

This is an active, data-driven iteration — not a one-off launch.

---

## Running Locally

**Backend:**
```bash
cd backend
pip install -r requirements.txt
export GOOGLE_API_KEY=your_key_here
uvicorn main:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

---

## What I Learned

Building RoleScope AI end-to-end — from prompt design and JSON-structured LLM outputs, to a from-scratch similarity-matching model, to production deployment and real-user analytics — was my first experience owning a product across its full lifecycle: concept, build, ship, measure, iterate. The GA4 bounce-rate finding in particular reinforced a lesson I keep coming back to: shipping is the beginning of the feedback loop, not the end of it.

---

## Author

**Mehmet Onur Pirencioğlu** — Industrial Engineer, Data & Growth Analytics
[LinkedIn](https://linkedin.com/in/mehmet-onur-pirencioglu) · [GitHub](https://github.com/monurpirencioglu)
