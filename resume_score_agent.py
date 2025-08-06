import os
import json
from groq import Groq
from typing import List, Tuple

# Initialize the Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))  # Make sure to set this in your env or .streamlit/secrets.toml

def extract_skills_from_resume(resume_text: str) -> List[str]:
    """Use LLM to extract all relevant skills, even weakly mentioned, from resume."""
    prompt = f"""
You are an expert resume parser.

Extract a list of all technical and soft skills (both strong and weak mentions) from the following resume text.

Resume Text:
\"\"\"
{resume_text}
\"\"\"

Return the result as a Python list of strings.
"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=512,
    )

    # Try to parse the returned content safely
    content = response.choices[0].message.content.strip()
    try:
        skills = eval(content)
        if isinstance(skills, list):
            return [skill.lower().strip() for skill in skills]
    except:
        pass
    return []

def extract_required_skills_from_job(job_description: str) -> List[str]:
    """Extract required skills from job description using LLM."""
    prompt = f"""
You are an expert job analyst.

From the following job description, extract all the required skills and technologies.

Job Description:
\"\"\"
{job_description}
\"\"\"

Return the result as a Python list of strings.
"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=512,
    )

    content = response.choices[0].message.content.strip()
    try:
        skills = eval(content)
        if isinstance(skills, list):
            return [skill.lower().strip() for skill in skills]
    except:
        pass
    return []

def score_resume(resume_skills: List[str], required_skills: List[str]) -> Tuple[int, List[str]]:
    """Calculate resume score and missing skills."""
    resume_set = set(resume_skills)
    required_set = set(required_skills)

    matched_skills = resume_set.intersection(required_set)
    missing_skills = [skill for skill in required_skills if skill not in matched_skills]

    score = int((len(matched_skills) / max(len(required_skills), 1)) * 100)
    return score, missing_skills

def analyze_resume(resume_text: str, job_description: str) -> dict:
    """Main function to analyze resume against job description."""
    resume_skills = extract_skills_from_resume(resume_text)
    required_skills = extract_required_skills_from_job(job_description)
    score, missing_skills = score_resume(resume_skills, required_skills)

    return {
        "resume_score": score,
        "matched_skills": list(set(resume_skills).intersection(set(required_skills))),
        "missing_skills": missing_skills,
        "built_by": "Chantibabusambangi"
    }
