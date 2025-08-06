import spacy
from sentence_transformers import SentenceTransformer, util
from typing import TypedDict, List
import torch
import re
from difflib import SequenceMatcher

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")
# Load transformer model once
model = SentenceTransformer('all-MiniLM-L6-v2')

class ResumeInput(TypedDict):
    resume_text: str
    jd_text: str
    job_skills: List[str]

class ResumeOutput(ResumeInput):
    score: float
    missing_skills: List[str]
    reasoning: str

# 1. Normalization and Lemmatization
def normalize_and_lemmatize(text: str) -> str:
    doc = nlp(text.lower().replace("-", " ").replace("_", " "))
    return " ".join([token.lemma_ for token in doc if token.is_alpha])

# 2. Gentle Chunking
def chunk_resume(text: str) -> List[str]:
    raw_chunks = re.split(r"[•\n]", text)
    return [normalize_and_lemmatize(chunk) for chunk in raw_chunks if chunk.strip()]

# 3. Skill Synonyms Expansion
SKILL_SYNONYMS = {
    "machine learning": ["machine learning", "ml", "statistical learning", "predictive modeling"],
    "nlp": ["nlp", "natural language processing", "text analytics"],
    "sql": ["sql", "structured query language"],
    "python": ["python"],
    # Add more domain-specific synonyms as needed
}

def expand_skills(skills: List[str]) -> List[str]:
    expanded = set()
    for skill in skills:
        key = skill.lower()
        expanded.update(SKILL_SYNONYMS.get(key, [key]))
    return list(expanded)

# 4. Whole-Word Matching
def skill_in_resume(skill: str, resume_text: str) -> bool:
    return re.search(r"\b{}\b".format(re.escape(skill)), resume_text) is not None

# 5. Fuzzy Multi-Word Matching
def fuzzy_phrase_match(skill: str, text: str, threshold=0.8) -> bool:
    return SequenceMatcher(None, skill, text).ratio() > threshold

# 6. Extract Technical Skills Section for Focused Scanning
def extract_skills_section(resume: str) -> List[str]:
    match = re.search(r"technical skills.*?(?=\n[A-Z][^a-z])", resume, re.DOTALL | re.IGNORECASE)
    if match:
        section = match.group()
        return [normalize_and_lemmatize(section)]
    return []

def score_resume_vs_jd(inputs: ResumeInput, config=None) -> ResumeOutput:
    resume = inputs["resume_text"]
    jd = inputs["jd_text"]
    job_skills = inputs["job_skills"]

    if not resume.strip() or not jd.strip():
        return {**inputs, "score": 0.0, "missing_skills": job_skills, "reasoning": "Empty resume or JD provided."}
    if len(resume.split()) < 20 or len(jd.split()) < 20:
        return {**inputs, "score": 0.0, "missing_skills": job_skills, "reasoning": "Resume or JD too short to analyze meaningfully."}

    resume_norm = normalize_and_lemmatize(resume)
    jd_norm = normalize_and_lemmatize(jd)

    chunks = chunk_resume(resume)
    skills_sections = extract_skills_section(resume)
    chunks.extend(skills_sections)

    emb_resume = model.encode(resume_norm, convert_to_tensor=True)
    emb_jd = model.encode(jd_norm, convert_to_tensor=True)
    emb_chunks = model.encode(chunks, convert_to_tensor=True) if chunks else torch.empty((0, model.get_sentence_embedding_dimension()))

    score_val = float(util.cos_sim(emb_resume, emb_jd).item() * 100)

    missing = []
    normalized_job_skills = [normalize_and_lemmatize(skill) for skill in job_skills]
    expanded_skills = expand_skills(normalized_job_skills)
    expanded_skill_embeddings = model.encode(expanded_skills, convert_to_tensor=True)

    resume_comp = " ".join(chunks)

    for idx, skill in enumerate(expanded_skills):
        found = False

        if skill_in_resume(skill, resume_comp):
            found = True
        elif fuzzy_phrase_match(skill, resume_comp):
            found = True
        else:
            emb_skill = expanded_skill_embeddings[idx]
            sim_resume = util.cos_sim(emb_skill, emb_resume).item() if emb_resume is not None else 0
            sim_chunks = torch.max(util.cos_sim(emb_skill, emb_chunks)).item() if emb_chunks.size(0) > 0 else 0
            if sim_resume > 0.48 or sim_chunks > 0.48:
                found = True

        if not found:
            original_skill_idx = None
            for i, skill_orig in enumerate(normalized_job_skills):
                if skill in SKILL_SYNONYMS.get(skill_orig, [skill_orig]):
                    original_skill_idx = i
                    break
            if original_skill_idx is not None:
                orig_skill_name = job_skills[original_skill_idx]
                if orig_skill_name not in missing:
                    missing.append(orig_skill_name)

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

# Usage: resume_skill_match_agent = score_resume_vs_jd
