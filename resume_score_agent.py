from typing import TypedDict, List
import os
import json
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

# Initialize Groq LLM - llama3-8b-8192
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama3-8b-8192")

class ResumeInput(TypedDict):
    resume_text: str
    jd_text: str
    job_skills: List[str]

class ResumeOutput(ResumeInput):
    score: float
    missing_skills: List[str]
    reasoning: str

def score_resume_vs_jd_llm(inputs: ResumeInput, config=None) -> ResumeOutput:
    """
    Uses Groq LLM (llama3-8b-8192) to compare resume and JD,
    detect missing skills, and provide a similarity score + explanation.
    """

    resume = inputs["resume_text"].strip()
    jd = inputs["jd_text"].strip()
    job_skills = inputs["job_skills"]

    # Basic validation
    if not resume or not jd:
        return {**inputs, "score": 0.0, "missing_skills": job_skills, "reasoning": "Empty resume or JD provided."}
    if len(resume.split()) < 20 or len(jd.split()) < 20:
        return {**inputs, "score": 0.0, "missing_skills": job_skills, "reasoning": "Resume or JD too short to analyze meaningfully."}

    # Prepare the prompt for LLM with clear instructions
    prompt = f"""
You are a highly skilled recruiter assistant AI.

You receive a candidate's resume text, a job description, and a list of key job skills required.

Task:
1. Analyze the resume and job description.
2. For each skill in the job skills list, determine if the candidate has that skill sufficiently.
3. Return a JSON object with these fields:
    - "score": a float percentage score (0-100) showing overall resume match to the JD.
    - "missing_skills": a list of skills NOT clearly demonstrated in the resume.
    - "reasoning": a concise explanation of the assessment.

Inputs:
---
Resume:
{resume}

Job Description:
{jd}

Job Skills:
{job_skills}

Please respond ONLY with the JSON object, no other text.
"""

    # Call LLM - you can adjust max_tokens, temperature, etc. if needed
    response = llm.invoke([HumanMessage(content=prompt)])
    llm_output = response.content.strip()

    # Attempt to parse JSON from LLM response
    try:
        parsed = json.loads(llm_output)
    except json.JSONDecodeError:
        # If parsing fails, return fallback with info
        return {
            **inputs,
            "score": 0.0,
            "missing_skills": job_skills,
            "reasoning": f"Failed to parse LLM output JSON. Raw output: {llm_output[:300]}..."
        }

    # Validate keys and types in parsed output
    score = parsed.get("score", 0.0)
    missing_skills = parsed.get("missing_skills", job_skills if score <= 15 else [])
    reasoning = parsed.get("reasoning", "")

    # Defensive data cleanup
    if not isinstance(missing_skills, list):
        missing_skills = job_skills if score <= 15 else []

    if not isinstance(score, (int, float)) or score < 0 or score > 100:
        score = 0.0

    if not isinstance(reasoning, str):
        reasoning = ""

    # Return the output dictionary consistent with ResumeOutput
    return {
        **inputs,
        "score": round(float(score), 2),
        "missing_skills": missing_skills,
        "reasoning": reasoning.strip()
    }

# Usage for Langchain or standalone
resume_skill_match_agent = score_resume_vs_jd_llm

