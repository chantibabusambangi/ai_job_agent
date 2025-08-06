import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq

load_dotenv()

# Setup LLM
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a resume evaluator. Be strict and professional."),
    ("human", """
You will be given a resume and a job description.

Your job is to:
1. Carefully compare the resume against the job description.
2. Identify **missing or weakly present skills**.
3. Provide a **score out of 100** based on the relevance of the resume to the JD.

Return JSON like this:
{
  "score": 74,
  "missing_skills": ["Time series forecasting", "PyTorch", "CI/CD"]
}

Be strict. If a skill is mentioned only vaguely, treat it as missing.

Resume:
---------
{resume}

Job Description:
----------------
{jd}
""")
])

parser = JsonOutputParser()

chain = prompt | llm | parser

def evaluate_resume(resume_text: str, jd_text: str) -> dict:
    try:
        result = chain.invoke({
            "resume": resume_text,
            "jd": jd_text
        })
        return result
    except Exception as e:
        print("❌ Error while scoring resume:", e)
        return {"score": 0, "missing_skills": ["Error during evaluation"]}

# Example usage
if __name__ == "__main__":
    resume = open("resume.txt", "r", encoding="utf-8").read()
    jd = open("job_description.txt", "r", encoding="utf-8").read()

    print("🔍 Evaluating Resume...")
    result = evaluate_resume(resume, jd)
    
    print("✅ Score:", result["score"])
    print("❌ Missing Skills:", result["missing_skills"])
