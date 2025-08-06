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
    This version has stronger prompt logic to reduce false negatives.
    """

    resume = inputs["resume_text"].strip()
    jd = inputs["jd_text"].strip()
    job_skills = inputs["job_skills"]

    # Basic validation
    if not resume or not jd:
        return {**inputs, "score": 0.0, "missing_skills": job_skills, "reasoning": "Empty resume or JD provided."}
    if len(resume.split()) < 20 or len(jd.split()) < 20:
        return {**inputs, "score": 0.0, "missing_skills": job_skills, "reasoning": "Resume or JD too short to analyze meaningfully."}

    # Strong prompt to explicitly require ALL missing skills listed
    prompt = f"""
You are a precise and strict recruitment assistant AI.

Given a candidate's resume text, a job description, and a list of key job skills required, your task is:

1. Analyze carefully if each skill in the job skills list is present in the resume with sufficient evidence.
2. List ALL missing or insufficient skills explicitly, even if partially missing or implied weakly.
3. Provide an overall resume match score between 0 and 100 percent.
4. Give a concise, clear reasoning explaining which skills are missing and why.

Please RESPOND ONLY with a valid JSON EXACTLY matching this format (no extra text or explanations):

{{
    "score": float,          // 0.0 - 100.0, overall fit percentage
    "missing_skills": [      // list of missing or insufficient skills (as given in job skills list)
        "skill1",
        "skill2"
    ],
    "reasoning": string      // concise explanation of assessment
}}

Use the following INPUTS:

Resume:
{resume}

Job Description:
{jd}

Job Skills:
{job_skills}

Reminders:
- If all skills are present, missing_skills must be an empty list [].
- Do NOT omit or skip any skills if they are missing or weak.
- Do NOT add skills not in the provided list.
- Be concise and factual.
"""

    # Call LLM - you may tune parameters if supported
    response = llm.invoke([HumanMessage(content=prompt)])
    llm_output = response.content.strip()

    # For debugging: uncomment to see raw output
    # print("LLM raw output:", llm_output)

    # Attempt to parse JSON from LLM response robustly
    try:
        parsed = json.loads(llm_output)
    except json.JSONDecodeError:
        # Sometimes LLM may produce non-JSON output; try to find JSON substring or fallback
        import re
        json_match = re.search(r"\{.*\}", llm_output, flags=re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
            except Exception:
                parsed = None
        else:
            parsed = None
        if not parsed:
            return {
                **inputs,
                "score": 0.0,
                "missing_skills": job_skills,
                "reasoning": f"Failed to parse LLM output JSON. Raw output begins with: {llm_output[:300]}..."
            }

    # Validate parsed data and fallback defaults
    score = parsed.get("score")
    missing_skills = parsed.get("missing_skills")
    reasoning = parsed.get("reasoning", "")

    if not isinstance(score, (int, float)) or not (0 <= score <= 100):
        score = 0.0  # fallback

    if not isinstance(missing_skills, list):
        missing_skills = job_skills  # pessimistic fallback

    # Ensure missing_skills only contains skills from job_skills list (normalize casing)
    valid_skills_set = set(map(str.lower, job_skills))
    filtered_missing_skills = []
    for skill in missing_skills:
        if isinstance(skill, str) and skill.lower() in valid_skills_set:
            filtered_missing_skills.append(skill)
    missing_skills = filtered_missing_skills if filtered_missing_skills else []

    # Return structured output
    return {
        **inputs,
        "score": round(float(score), 2),
        "missing_skills": missing_skills,
        "reasoning": reasoning.strip()
    }

# Usage for Langchain or standalone
resume_skill_match_agent = score_resume_vs_jd_llm
