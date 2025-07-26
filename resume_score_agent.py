from sentence_transformers import SentenceTransformer, util
from typing import TypedDict, List
from langchain_core.runnables import RunnableConfig, RunnableLambda
import torch
import nltk
from nltk.tokenize import sent_tokenize
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

def normalize_skill(skill: str) -> str:
    return re.sub(r"[^\w\s]", "", skill.lower().replace("-", " ").replace("_", " ").strip())

def score_resume_vs_jd(inputs: ResumeInput, config: RunnableConfig = None) -> ResumeOutput:
    resume = inputs["resume_text"]
    jd = inputs["jd_text"]
    job_skills = inputs["job_skills"]

    # Basic input validation
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

    # Preprocess resume and JD
    jd_cleaned = jd.lower().replace("-", " ").replace("_", " ").strip()
    resume_chunks = [
        re.sub(r"[^\w\s]", "", chunk.lower().replace("-", " ").replace("_", " ").strip())
        for chunk in sent_tokenize(resume)
    ]

    # Embeddings
    emb_resume = model.encode(resume, convert_to_tensor=True)
    emb_jd = model.encode(jd_cleaned, convert_to_tensor=True)
    emb_chunks = model.encode(resume_chunks, convert_to_tensor=True)

    # Compute similarity score
    score = float(util.cos_sim(emb_resume, emb_jd).item() * 100)

    # Missing skills detection
    normalized_job_skills = [normalize_skill(skill) for skill in job_skills]
    skill_embeddings = model.encode(normalized_job_skills, convert_to_tensor=True)

    missing_skills = []
    for i, skill_emb in enumerate(skill_embeddings):
        sim_scores = util.cos_sim(skill_emb, emb_chunks)
        if torch.max(sim_scores).item() < 0.55:
            missing_skills.append(job_skills[i])

    # Reasoning
    if score > 75:
        reasoning = "High similarity indicates good alignment with the JD."
    elif score > 50:
        reasoning = "Moderate alignment. Some key skills may be missing or weakly represented."
    else:
        reasoning = "Low alignment. Resume may not be a good fit for the JD."

    return {
        **inputs,
        "score": round(score, 2),
        "missing_skills": missing_skills,
        "reasoning": reasoning
    }

# Agent wrapper
resume_skill_match_agent = RunnableLambda(score_resume_vs_jd)
