import os
from groq import Groq
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = Groq(api_key=GROQ_API_KEY)

def query_llm(prompt: str) -> str:
    response = llm.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024
    )
    return response.choices[0].message.content.strip()

def get_resume_score_and_missing_skills(resume_text: str, job_description: str):
    prompt = f"""
You are a professional career assistant helping candidates improve their resumes.

Compare the following RESUME and JOB DESCRIPTION.

1. Score the resume out of 100 based on:
   - Skill match
   - Relevance
   - Clarity
   - Technical depth

2. Identify **missing or weakly represented skills** required in the job description but not well covered in the resume. Be strict – if a skill is only briefly mentioned, still list it.

3. Return your response in the following format:

Resume Score: <score>/100

Missing Skills:
- <Skill 1>
- <Skill 2>
...

--- RESUME START ---
{resume_text}
--- RESUME END ---

--- JOB DESCRIPTION START ---
{job_description}
--- JOB DESCRIPTION END ---
"""
    try:
        response = query_llm(prompt)
        return response
    except Exception as e:
        return f"Error occurred: {e}"
