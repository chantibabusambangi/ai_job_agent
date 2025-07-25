from sentence_transformers import SentenceTransformer, util
from typing import TypedDict, List
from langchain_core.runnables import RunnableConfig
import torch

model = SentenceTransformer('all-MiniLM-L6-v2')

class ResumeInput(TypedDict):
    resume_text: str
    jd_text: str
    job_skills: List[str]

class ResumeOutput(ResumeInput):
    score: float
    missing_skills: List[str]
    reasoning: str

def score_resume_vs_jd(inputs: ResumeInput, config: RunnableConfig = None) -> ResumeOutput:
    resume = inputs["resume_text"]
    jd = inputs["jd_text"]
    job_skills = inputs["job_skills"]

    emb_resume = model.encode(resume, convert_to_tensor=True)
    emb_jd = model.encode(jd, convert_to_tensor=True)

    score = float(util.cos_sim(emb_resume, emb_jd).item() * 100)

    emb_skills = model.encode(job_skills, convert_to_tensor=True)
    skill_sims = util.cos_sim(emb_skills, emb_resume)  # shape: (len(skills), 1)

    missing_skills = [
        skill for i, skill in enumerate(job_skills)
        if skill_sims[i].item() < 0.6
    ]

    reasoning = (
        "High similarity indicates good alignment with the JD." if score > 75 else
        "Moderate alignment. Some key skills may be missing or weakly represented." if score > 50 else
        "Low alignment. Resume may not be a good fit for the JD."
    )

    return {
        **inputs,
        "score": round(score, 2),
        "missing_skills": missing_skills,
        "reasoning": reasoning
    }

