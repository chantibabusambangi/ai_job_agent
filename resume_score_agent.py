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
from typing import List as _List

def smart_chunk_resume(text: str) -> _List[str]:
    # Split on bullets, new lines, commas, semicolons, colons, and periods
    chunks = re.split(r'[•\n,;:.]', text)
    return [chunk.strip().lower() for chunk in chunks if chunk.strip()]

# ✅ Optional: Extract 'Technical Skills' section if present

def extract_technical_skills(resume: str) -> _List[str]:
    # Regex to find a section starting with 'Technical Skills' or 'Skills'
    match = re.search(r"(?i)(?:technical\s+skills|skills)\s*[:\-]?.*?(?=\n[A-Z][^a-z])", resume, re.DOTALL)
    if match:
        section = match.group()
        # Split by commas, newlines or bullets
        items = re.split(r'[:,\n•]', section)
        return [item.strip().lower() for item in items if item.strip()]
    return []

# ✅ Normalize skill keywords

def normalize_skill(skill: str) -> str:
    return re.sub(r"[^\w\s]", "", skill.lower().replace("-", " ").replace("_", " ").strip())

# ✅ Main scoring function

def score_resume_vs_jd(inputs: ResumeInput, config: RunnableConfig = None) -> ResumeOutput:
    resume = inputs["resume_text"]
    jd = inputs["jd_text"]
    job_skills = inputs["job_skills"]

    # Validation
    if not resume.strip() or not jd.strip():
        return { **inputs, "score": 0.0, "missing_skills": job_skills, "reasoning": "Empty resume or JD provided." }
    if len(resume.split()) < 20 or len(jd.split()) < 20:
        return { **inputs, "score": 0.0, "missing_skills": job_skills, "reasoning": "Resume or JD too short to analyze meaningfully." }

    # Preprocess JD
    jd_cleaned = jd.lower().replace("-", " ").replace("_", " ").strip()

    # Chunk resume text
    resume_chunks = smart_chunk_resume(resume)
    tech_chunks = extract_technical_skills(resume)
    combined_chunks = resume_chunks + tech_chunks

    # Clean chunks: remove punctuation for embedding consistency
    cleaned_chunks = [ re.sub(r"[^\w\s]", "", chunk) for chunk in combined_chunks ]

    # Compute embeddings
    emb_resume = model.encode(resume, convert_to_tensor=True)
    emb_jd = model.encode(jd_cleaned, convert_to_tensor=True)
    emb_chunks = model.encode(cleaned_chunks, convert_to_tensor=True)

    # Compute overall similarity score
    score_val = float(util.cos_sim(emb_resume, emb_jd).item() * 100)

    # Missing skills detection with fallback keyword match
    normalized_job_skills = [normalize_skill(skill) for skill in job_skills]
    skill_embeddings = model.encode(normalized_job_skills, convert_to_tensor=True)

    resume_text_lower = resume.lower()
    missing = []

    for idx, skill_emb in enumerate(skill_embeddings):
        skill_key = normalized_job_skills[idx]
        # Fallback: direct substring check
        if skill_key in resume_text_lower:
            continue
        # Semantic match check
        sim_scores = util.cos_sim(skill_emb, emb_chunks)
        if torch.max(sim_scores).item() < 0.55:
            missing.append(job_skills[idx])

    # Interpret score
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

# Agent wrapper
resume_skill_match_agent = RunnableLambda(score_resume_vs_jd)
