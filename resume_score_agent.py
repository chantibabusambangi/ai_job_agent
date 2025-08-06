import re
from typing import TypedDict, List
from difflib import SequenceMatcher
import spacy
import torch
from sentence_transformers import SentenceTransformer, util

# Load spaCy model and SentenceTransformer model once
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')

class ResumeInput(TypedDict):
    resume_text: str
    jd_text: str
    job_skills: List[str]

class ResumeOutput(ResumeInput):
    score: float
    missing_skills: List[str]
    reasoning: str

# --- Utility Functions ---

def normalize_and_lemmatize(text: str) -> str:
    """Lowercase, replace separators, remove punctuation, and lemmatize text."""
    text = text.lower().replace("-", " ").replace("_", " ")
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if token.is_alpha]
    return " ".join(lemmas)

def gentle_chunk_resume(text: str) -> List[str]:
    """Split resume on newlines or bullet points & normalize."""
    raw_chunks = re.split(r"[•\n]", text)
    return [normalize_and_lemmatize(chunk) for chunk in raw_chunks if chunk.strip()]

def skill_in_resume(skill: str, resume_text: str) -> bool:
    """
    Regex whole-word match that allows optional punctuation/whitespace between skill words.
    E.g., matches "machine-learning", "machine learning", "machine, learning".
    """
    skill_words = skill.split()
    pattern = r"\b" + r"\W*".join(map(re.escape, skill_words)) + r"\b"
    return re.search(pattern, resume_text, flags=re.IGNORECASE) is not None

def fuzzy_phrase_match(skill: str, text: str, threshold: float = 0.8) -> bool:
    """Use difflib to return True if similarity ratio exceeds the threshold."""
    return SequenceMatcher(None, skill, text).ratio() > threshold

def extract_technical_skills_section(resume_text: str) -> List[str]:
    """
    Extract the 'Technical Skills' section or 'Skills' section from resume text.
    Returns list of lemmatized normalized chunks for focused skill matching.
    """
    # Regex to capture from 'Technical Skills' or 'Skills' heading up to next heading or end
    match = re.search(r"(?i)(technical skills|skills)[\s:\-]*([\s\S]+?)(?=\n[A-Z][a-z]|$)", resume_text)
    if match:
        section_text = match.group(2)
        # Split on commas, new lines, bullets and normalize
        raw_skills = re.split(r"[,•\n]", section_text)
        return [normalize_and_lemmatize(s) for s in raw_skills if s.strip()]
    return []

# Synonyms dictionary: map canonical skill → list of synonyms to check for
SKILL_SYNONYMS = {
    "machine learning": ["machine learning", "ml", "statistical learning", "predictive modeling"],
    "nlp": ["natural language processing", "nlp", "text analytics", "text analysis"],
    "sql": ["sql", "structured query language"],
    "python": ["python"],
    # Add more synonyms as relevant for your domain
}

def expand_skills(skills: List[str]) -> List[str]:
    """Expand skills list with synonyms if available."""
    expanded = set()
    for skill in skills:
        key = skill.lower()
        synonyms = SKILL_SYNONYMS.get(key, [key])
        expanded.update(synonyms)
    return list(expanded)

def score_resume_vs_jd(inputs: ResumeInput, config=None) -> ResumeOutput:
    resume = inputs["resume_text"]
    jd = inputs["jd_text"]
    job_skills = inputs["job_skills"]

    # Basic validation
    if not resume.strip() or not jd.strip():
        return {
            **inputs, 
            "score": 0.0, 
            "missing_skills": job_skills, 
            "reasoning": "Empty resume or JD provided."
        }
    if len(resume.split()) < 20 or len(jd.split()) < 20:
        return {
            **inputs,
            "score": 0.0,
            "missing_skills": job_skills,
            "reasoning": "Resume or JD too short to analyze meaningfully."
        }

    # Normalize and lemmatize whole resume and JD
    resume_norm = normalize_and_lemmatize(resume)
    jd_norm = normalize_and_lemmatize(jd)

    # Chunk resume gently & extract skills section for focused matching
    chunks = gentle_chunk_resume(resume)
    skills_section_chunks = extract_technical_skills_section(resume)
    chunks.extend(skills_section_chunks)

    # Compute embeddings
    emb_resume = model.encode(resume_norm, convert_to_tensor=True)
    emb_jd = model.encode(jd_norm, convert_to_tensor=True)
    emb_chunks = model.encode(chunks, convert_to_tensor=True) if chunks else torch.empty((0, model.get_sentence_embedding_dimension()))

    # Overall resume-JD similarity score (0-100)
    score_val = float(util.cos_sim(emb_resume, emb_jd).item() * 100)

    # Skill checking
    missing = []
    normalized_job_skills = [normalize_and_lemmatize(skill) for skill in job_skills]
    expanded_skills = expand_skills(normalized_job_skills)
    emb_expanded_skills = model.encode(expanded_skills, convert_to_tensor=True)

    # Combine all chunks for lexical & fuzzy matching to a single string for convenience
    combined_chunks_text = " ".join(chunks)

    for idx, skill in enumerate(expanded_skills):
        found = False

        # 1. Whole-word regex match against combined chunks text
        if skill_in_resume(skill, combined_chunks_text):
            found = True
        # 2. Fuzzy phrase match against combined chunks text
        elif fuzzy_phrase_match(skill, combined_chunks_text):
            found = True
        else:
            # 3. Semantic similarity fallback on resume embedding and chunk embeddings
            emb_skill = emb_expanded_skills[idx]
            sim_resume = util.cos_sim(emb_skill, emb_resume).item()
            sim_chunks = torch.max(util.cos_sim(emb_skill, emb_chunks)).item() if emb_chunks.size(0) > 0 else 0
            if sim_resume > 0.48 or sim_chunks > 0.48:
                found = True

        if not found:
            # Map expanded skill back to original job skill
            mapped_original_skill = None
            for orig_idx, orig_skill_norm in enumerate(normalized_job_skills):
                if skill in SKILL_SYNONYMS.get(orig_skill_norm, [orig_skill_norm]):
                    mapped_original_skill = job_skills[orig_idx]
                    break
            if mapped_original_skill and mapped_original_skill not in missing:
                missing.append(mapped_original_skill)

    # Reasoning based on score
    if score_val > 40:
        reasoning = "✅ High similarity — resume aligns well with the job description."
    elif score_val > 20:
        reasoning = "⚠️ Moderate similarity — resume partially aligns; some skills may be missing."
    else:
        reasoning = "❌ Low similarity — resume lacks significant alignment with the JD."

    return {
        **inputs,
        "score": round(score_val, 2),
        "missing_skills": missing,
        "reasoning": reasoning,
    }

# The agent function for external usage
resume_skill_match_agent = score_resume_vs_jd

