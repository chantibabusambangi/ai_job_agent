from sentence_transformers import SentenceTransformer, util
from typing import TypedDict
from langchain_core.runnables import RunnableConfig
import torch

# Load model once globally
model = SentenceTransformer('all-MiniLM-L6-v2')

# Input/Output schema
class ResumeInput(TypedDict):
    resume_text: str
    jd_text: str
    job_skills: list[str]

class ResumeOutput(ResumeInput):
    resume_score: float
    missing_skills: list[str]
    reasoning: str

def score_resume_vs_jd(inputs: ResumeInput, config: RunnableConfig = None) -> ResumeOutput:
    resume = inputs["resume_text"]
    jd = inputs["jd_text"]
    job_skills = inputs["job_skills"]

    # Encode resume and JD
    emb_resume = model.encode(resume, convert_to_tensor=True)
    emb_jd = model.encode(jd, convert_to_tensor=True)

    # Compute overall similarity score
    score = float(util.cos_sim(emb_resume, emb_jd).item() * 100)

    # Identify missing skills
    missing_skills = []
    for skill in job_skills:
        emb_skill = model.encode(skill, convert_to_tensor=True)
        similarity = util.cos_sim(emb_skill, emb_resume).item()
        if similarity < 0.6:
            missing_skills.append(skill)

    # Reasoning logic
    if score > 75:
        reasoning = "High similarity indicates good alignment with the JD."
    elif score > 50:
        reasoning = "Moderate alignment. Some key skills may be missing or weakly represented."
    else:
        reasoning = "Low alignment. Resume may not be a good fit for the JD."

    return {
        **inputs,
        "resume_score": round(score, 2),
        "missing_skills": missing_skills,
        "reasoning": reasoning
    }
