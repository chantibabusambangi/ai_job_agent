from crewai import Agent, Task, Crew
from textwrap import dedent
import os

# You can replace this with any source of required skills
REQUIRED_SKILLS = [
    "Python", "TensorFlow", "PyTorch", "Scikit-learn", "Deep Learning",
    "Machine Learning", "NLP", "CNN", "Data Analysis", "NumPy", "pandas",
    "Matplotlib", "Transformer", "LLM", "AutoEncoder", "PCA", "GAN", "OpenCV",
    "FastAPI", "Docker", "SQL", "Git"
]

def resume_score_agent(resume_text):
    from langchain.llms import Groq
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain

    llm = Groq(temperature=0.3, model_name="llama3-8b-8192")

    # Prompt to extract skills and compare
    prompt = PromptTemplate(
        input_variables=["resume_text", "required_skills"],
        template=dedent("""
        You are an expert resume evaluator.

        Below is a resume:
        -----------------
        {resume_text}
        -----------------

        The required skills for the role are:
        {required_skills}

        Your job is to:
        1. Extract all clearly and weakly mentioned skills from the resume.
        2. Compare them with the required skills.
        3. Return:
            - A list of missing skills (include even if mentioned weakly).
            - A resume score out of 10 based on relevance to required skills.

        Respond in the following JSON format:

        {{
            "extracted_skills": [...],
            "missing_skills": [...],
            "resume_score": x
        }}
        """)
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    result = chain.run({
        "resume_text": resume_text,
        "required_skills": ", ".join(REQUIRED_SKILLS)
    })

    # Try to extract dictionary safely
    import json
    try:
        json_result = json.loads(result)
        return {
            "extracted_skills": json_result.get("extracted_skills", []),
            "missing_skills": json_result.get("missing_skills", []),
            "resume_score": json_result.get("resume_score", 0)
        }
    except json.JSONDecodeError:
        # fallback: handle malformed LLM response
        return {
            "extracted_skills": [],
            "missing_skills": REQUIRED_SKILLS,
            "resume_score": 0
        }
from resume_score_agent import resume_score_agent

resume_text = your_resume_text_input
result = resume_score_agent(resume_text)

st.write("✅ Extracted Skills:", result["extracted_skills"])
st.write("❌ Missing Skills:", result["missing_skills"])
st.write("📊 Resume Score:", result["resume_score"], "/10")
