import streamlit as st
import os
from openai import OpenAI
from typing import List

# Streamlit App Title
st.title("🔍 Resume Skill Matcher")
st.caption("Built by chantibabusambangi")

# Load LLM from Groq (assuming OpenAI compatibility)
llm = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",  # adjust if needed
)

# Helper to get missing skills using LLM
def get_missing_skills_llm(resume_text: str, job_description: str) -> List[str]:
    prompt = f"""
You are an expert in resume evaluation. Your job is to extract a list of **missing skills** from a candidate’s resume based on a given job description. 

Your job is to:
- Compare job description and resume
- Consider a skill *missing* even if it is weakly mentioned or poorly demonstrated
- Return ONLY the skills missing or weakly present in the resume

### Job Description:
{job_description}

### Resume:
{resume_text}

### Output (comma-separated skills missing):
"""

    try:
        response = llm.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        content = response.choices[0].message.content
        # Parse result assuming LLM outputs comma-separated values
        missing_skills = [skill.strip() for skill in content.split(",") if skill.strip()]
        return missing_skills
    except Exception as e:
        st.error(f"Error calling LLM: {e}")
        return []

# UI Inputs
resume_text = st.text_area("Paste your resume content here", height=300)
job_description = st.text_area("Paste the job description here", height=300)

# Process Button
if st.button("Analyze Missing Skills"):
    if not resume_text or not job_description:
        st.warning("Please provide both resume and job description.")
    else:
        with st.spinner("Analyzing..."):
            missing_skills = get_missing_skills_llm(resume_text, job_description)
        if not missing_skills:
            st.success("✅ No significant missing skills found. Good job!")
        else:
            st.error("🚫 Missing or weakly demonstrated skills:")
            for skill in missing_skills:
                st.write(f"- {skill}")
