import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Groq client
llm = Groq(api_key=GROQ_API_KEY)

# LangGraph-compatible node function
def resume_skill_match_agent(state: dict) -> dict:
    resume_text = state.get("resume_text", "")
    jd_text = state.get("jd_text", "")
    job_skills = state.get("job_skills", [])

    if not resume_text or not jd_text or not job_skills:
        return {
            "score": 0.0,
            "missing_skills": job_skills,
            "reasoning": "Missing input data."
        }

    # Prepare skills as bullet list
    skills_bulleted = "\n".join(f"- {s}" for s in job_skills)

    prompt = f"""
You are an expert technical recruiter and resume evaluator.

Your task is to:
1. Read the RESUME and JOB DESCRIPTION.
2. Compare the resume against the required skills.
3. Give a score out of 100 indicating how well the resume matches the job.
4. List all missing or weakly mentioned skills (even if mentioned only once or without detail).

Return result in this JSON format only (no explanation):

{{
  "score": <resume_score>,
  "missing_skills": [<list_of_missing_skills_as_strings>]
}}

--- RESUME START ---
{resume_text}
--- RESUME END ---

--- JOB DESCRIPTION START ---
{jd_text}
--- JOB DESCRIPTION END ---

--- JOB SKILLS ---
{skills_bulleted}
--- END SKILLS ---
"""

    try:
        response = llm.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024
        )
        content = response.choices[0].message.content.strip()

        import json
        parsed = json.loads(content)

        # Validate and return safely
        return {
            "score": float(parsed.get("score", 0.0)),
            "missing_skills": parsed.get("missing_skills", [])
        }

    except Exception as e:
        return {
            "score": 0.0,
            "missing_skills": job_skills,
            "reasoning": f"LLM error: {e}"
        }
