from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

# System prompt to define agent’s behavior
system_template = """
You are an expert career coach AI. 
Given a candidate's resume and a job description, your job is to:
1. Extract relevant skills from the resume and job description.
2. Match them to compute a resume-job fit score out of 100.
3. Identify and list missing skills (i.e., required in JD but missing in resume).
Be fair, thorough, and actionable in your analysis.
"""

# Create a prompt for the LLM
prompt_template = PromptTemplate(
    template="""
    [RESUME]:
    {resume_text}

    [JOB DESCRIPTION]:
    {jd_text}

    Please perform the following:
    - Extract the top relevant skills from both documents.
    - Compute a similarity score (0 to 100) based on skills match.
    - List all missing skills.
    Respond in JSON format:
    {{
        "score": <int>,
        "missing_skills": [<skill_1>, <skill_2>, ...],
        "analysis": "<summary of the reasoning>"
    }}
    """,
    input_variables=["resume_text", "jd_text"]
)

# Define the chain using LangChain agent-like behavior
llm = ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo")
chain = LLMChain(llm=llm, prompt=prompt_template)

# Define the actual runnable node
def resume_skill_match_agent(state):
    resume_text = state["resume_text"]
    jd_text = state["jd_text"]

    # Run chain with inputs
    response = chain.run({
        "resume_text": resume_text,
        "jd_text": jd_text
    })

    # Optional: convert string response to dict
    import json
    try:
        result_json = json.loads(response)
    except Exception:
        result_json = {
            "score": 0,
            "missing_skills": [],
            "analysis": "Unable to parse response."
        }

    # Return updated state (agentic behavior: communicate with others)
    return {
        **state,
        "score": result_json["score"],
        "missing_skills": result_json["missing_skills"],
        "analysis": result_json["analysis"],
        "result": f"Resume Score: {result_json['score']}/100\n\nMissing Skills: {', '.join(result_json['missing_skills']) if result_json['missing_skills'] else 'None'}\n\nAnalysis: {result_json['analysis']}"
    }

