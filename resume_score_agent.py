import os
import re
import openai  # or groq if using their client
import streamlit as st
from typing import List
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Set up LLaMA3 with Groq
openai.api_key = GROQ_API_KEY
openai.api_base = "https://api.groq.com/openai/v1"  # Groq endpoint

llm_model = "llama3-8b-8192"

def call_llm(prompt: str) -> str:
    response = openai.ChatCompletion.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant that analyzes resumes."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    return response['choices'][0]['message']['content'].strip()

def extract_resume_skills(resume_text: str, skill_list: List[str]) -> dict:
    skills_str = ", ".join(skill_list)

    prompt = f"""
    You are given a resume and a list of target skills.

    TASK:
    - Identify which skills are clearly or weakly present in the resume.
    - Return two comma-separated lists:
      1. `present_skills`: skills that are either clearly or weakly present.
      2. `missing_skills`: skills that are missing or not found at all.
    - Include weak mentions (e.g. in projects or education) in present_skills.
    - Only return the response in this JSON format:
      {{
        "present_skills": [...],
        "missing_skills": [...]
      }}

    Resume:
    {resume_text}

    Skills to match:
    [{skills_str}]
    """

    try:
        llm_output = call_llm(prompt)
        parsed = eval(llm_output) if llm_output.startswith("{") else {}
        return {
            "present_skills": parsed.get("present_skills", []),
            "missing_skills": parsed.get("missing_skills", [])
        }
    except Exception as e:
        st.error(f"Error parsing LLM response: {e}")
        return {
            "present_skills": [],
            "missing_skills": skill_list
        }

def score_resume(present_skills: List[str], total_skills: List[str]) -> float:
    if not total_skills:
        return 0.0
    score = (len(present_skills) / len(total_skills)) * 100
    return round(score, 2)

# This can be used in Streamlit like:
# result = extract_resume_skills(resume_text, skill_list)
# score = score_resume(result["present_skills"], skill_list)



