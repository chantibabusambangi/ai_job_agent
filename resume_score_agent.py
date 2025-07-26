from sentence_transformers import SentenceTransformer, util
from typing import TypedDict, List
from langchain_core.runnables import RunnableConfig, RunnableLambda
import torch
from nltk.tokenize import sent_tokenize

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

    def normalize_skill(skill: str) -> str:
        return skill.lower().replace("-", " ").replace("_", " ").strip()

    # Normalize and embed resume and JD
    jd = jd.lower().replace("-", " ").replace("_", " ").strip()
    emb_resume = model.encode(resume, convert_to_tensor=True)
    emb_jd = model.encode(jd, convert_to_tensor=True)

    score = float(util.cos_sim(emb_resume, emb_jd).item() * 100)

    
    resume_chunks = [
        chunk.lower().replace("-", " ").replace("_", " ").strip()
        for chunk in sent_tokenize(resume)
    ]

    emb_chunks = model.encode(resume_chunks, convert_to_tensor=True)

    missing_skills = []
    normalized_job_skills = [normalize_skill(skill) for skill in job_skills]
    for original_skill, normalized_skill in zip(job_skills, normalized_job_skills):
        skill_emb = model.encode(normalized_skill, convert_to_tensor=True)
        sim_scores = util.cos_sim(skill_emb, emb_chunks)
        max_sim = torch.max(sim_scores).item()
        if max_sim < 0.55:
            missing_skills.append(original_skill)


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

resume_skill_match_agent = RunnableLambda(score_resume_vs_jd)
