from sentence_transformers import SentenceTransformer, util
from typing import TypedDict, List
from langchain_core.runnables import RunnableConfig, RunnableLambda
import torch
import re

# Load model once
model = SentenceTransformer('all-MiniLM-L6-v2')

class ResumeInput(TypedDict):
    resume_text: str
    jd_text: str
    job_skills: List[str]

class ResumeOutput(ResumeInput):
    score: float
    missing_skills: List[str]
    reasoning: str

# ✅ Improved chunking
def smart_chunk_resume(text: str) -> List[str]:
    chunks = re.split(r'[•\n,;:.]', text)  # Bullet, newline, comma, colon, etc.
    return [chunk.strip().lower() for chunk in chunks if chunk.strip()]

# ✅ Optional: Extract 'Technical Skills' block
def extract_technical_skills(resume: str) -> List[str]:
    match = re.search(r"(?i)technical\s+skills.*?(?=\n[A-Z][^a-z])", resume, re.DOTALL)
    if match:
        return re.split(r"[:,\n•]", match.group())
    return []

# ✅ Normalize skills
def normalize_skill(skill: str) -> str:
    return re.sub(r"[^\w\s]", "", skill.lower().replace("-", " ").replace("_", " ").strip())

# ✅ Main function
def score_resume_vs_jd(inputs: ResumeInput, config: RunnableConfig = None) -> ResumeOutput:
    resume = inputs["resume_text"]
    jd = inputs["jd_text"]
    job_skills = inputs["job_skills"]

    if not resume.strip() or not jd.strip():
        return {
            **inputs,
            "score": 0.0,
            "missing_skills": job_skills,
            "reasoning": "Empty resume or JD provided."
        }
    if len(resume.strip().split()) < 20 or len(jd.strip().split()) < 20:
        return {
            **inputs,
            "score": 0.0,
            "missing_skills": job_skills,
            "reasoning": "Resume or JD too short to analyze meaningfully."
        }

    jd_cleaned = jd.lower().replace("-", " ").replace("_", " ").strip()

    # ✅ Combine chunking + optional skill section
    resume_chunks = smart_chunk_resume(resume)
    tech_skill_chunks = extract_technical_skills(resume)
    combined_chunks = resume_chunks + tech_skill_chunks

    cleaned_chunks = [
        re.sub(r"[^\w\s]", "", chunk.strip())
        for chunk in combined_chunks
        if chunk.strip()
    ]

    # Embeddings
    emb_resume = model.encode(resume, convert_to_tensor=True)
    emb_jd = model.encode(jd_cleaned, convert_to_tensor=True)
    emb_chunks = model.encode(cleaned_chunks, convert_to_tensor=True)

    # Cosine similarity
    score = float(util.cos_sim(emb_resume, emb_jd).item() * 100)

    # Missing skills check
    normalized_job_skills = [normalize_skill(skill) for skill in job_skills]
    skill_embeddings = model.encode(normalized_job_skills, convert_to_tensor=True)

    resume_text_lower = resume.lower()
    missing_skills = []

    for i, skill_emb in enumerate(skill_embeddings):
        skill_name = normalized_job_skills[i]

        if skill_name in resume_text_lower:
            continue  # Skill found directly

        sim_scores = util.cos_sim(skill_emb, emb_chunks)
        if torch.max(sim_scores).item() < 0.55:
            missing_skills.append(job_skills[i])

    # Score interpretation
    if score > 50:
        reasoning = "✅ High similarity — resume aligns well with the job description."
    elif score > 35:
        reasoning = "⚠️ Moderate similarity — resume partially aligns, but some important skills may be missing."
    else:
        reasoning = "❌ Low similarity — resume lacks significant alignment with the JD."

    return {
        **inputs,
        "score": round(score, 2),
        "missing_skills": missing_skills,
        "reasoning": reasoning
    }

# Agent wrapper
resume_skill_match_agent = RunnableLambda(score_resume_vs_jd)
