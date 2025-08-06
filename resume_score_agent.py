from sentence_transformers import SentenceTransformer, util
from typing import TypedDict, List
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

# Robust normalization
def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("-", " ").replace("_", " ")
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Less aggressive chunking: only on newlines and bullets
def chunk_resume(text: str) -> List[str]:
    return [normalize_text(chunk) for chunk in re.split(r"[•\n]", text) if chunk.strip()]

# Whole-word skill check using regex
def skill_in_resume(skill: str, resume_text: str) -> bool:
    pattern = r"\b{}\b".format(re.escape(skill))
    return re.search(pattern, resume_text) is not None

def score_resume_vs_jd(inputs: ResumeInput, config=None) -> ResumeOutput:
    resume = inputs["resume_text"]
    jd = inputs["jd_text"]
    job_skills = inputs["job_skills"]

    # Validation
    if not resume.strip() or not jd.strip():
        return { **inputs, "score": 0.0, "missing_skills": job_skills, "reasoning": "Empty resume or JD provided." }
    if len(resume.split()) < 20 or len(jd.split()) < 20:
        return { **inputs, "score": 0.0, "missing_skills": job_skills, "reasoning": "Resume or JD too short to analyze meaningfully." }

    # Normalization
    resume_normalized = normalize_text(resume)
    jd_normalized = normalize_text(jd)
    chunks = chunk_resume(resume)

    # Embeddings for semantic similarity
    emb_resume = model.encode(resume_normalized, convert_to_tensor=True)
    emb_jd = model.encode(jd_normalized, convert_to_tensor=True)
    emb_chunks = model.encode(chunks, convert_to_tensor=True)

    # Score between resume and JD
    score_val = float(util.cos_sim(emb_resume, emb_jd).item() * 100)

    # Prepare for skills extraction
    missing = []
    normalized_job_skills = [normalize_text(skill) for skill in job_skills]
    skill_embeddings = model.encode(normalized_job_skills, convert_to_tensor=True)

    # Check each skill
    for idx, skill in enumerate(normalized_job_skills):
        # 1. Whole-word lexical match on resume
        if skill_in_resume(skill, resume_normalized):
            continue

        # 2. Semantic similarity vs. resume chunks
        sim_chunk = util.cos_sim(skill_embeddings[idx], emb_chunks)
        sim_resume = util.cos_sim(skill_embeddings[idx], emb_resume)

        # If not found lexically and below threshold for both, mark as missing
        if torch.max(sim_chunk).item() < 0.55 and sim_resume.item() < 0.55:
            missing.append(job_skills[idx])  # append original form

    # Score interpretation
    if score_val > 75:
        reasoning = "✅ High similarity — resume aligns well with the job description."
    elif score_val > 50:
        reasoning = "⚠️ Moderate similarity — resume partially aligns; some skills may be missing."
    else:
        reasoning = "❌ Low similarity — resume lacks significant alignment with the JD."

    return {
        **inputs,
        "score": round(score_val, 2),
        "missing_skills": missing,
        "reasoning": reasoning
    }

# Usage (for Langchain or standalone)
resume_skill_match_agent = score_resume_vs_jd
