# resume_skill_match_agent.py
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_core.runnables import RunnableLambda
# from youtube_utility import suggest_youtube_channels 
import re

# Mock LLM
from langchain_groq import ChatGroq
import os

chat_model = ChatGroq(
    model="llama3-70b-8192",  # or llama3-70b-8192
    api_key=""
) 

#TOOLS

@tool
def extract_resume_skills(resume: str) -> List[str]:
    """Extract skills from a given resume"""
    prompt = f"List the key technical and soft skills in this resume:\n\n{resume}"
    response = chat_model.invoke([HumanMessage(content=prompt)])
    return parse_skills(response.content)

@tool
def extract_jd_skills(jd: str) -> List[str]:
    """Extract required skills from a job description"""
    prompt = f"List the required skills mentioned in this job description:\n\n{jd}"
    response = chat_model.invoke([HumanMessage(content=prompt)])
    return parse_skills(response.content)

@tool
def compare_skills(resume_skills: List[str], jd_skills: List[str]) -> dict:
    """Compare resume and JD skills, return missing and matched"""
    matched = list(set(resume_skills) & set(jd_skills))
    missing = list(set(jd_skills) - set(resume_skills))
    return {"matched": matched, "missing": missing}

@tool
def score_match(matched: List[str], jd_skills: List[str]) -> int:          
    """Score the match out of 100 based on JD skills"""
    if not jd_skills:
        return 0
    return int(len(matched) / len(jd_skills) * 100) #make this semantic using embeddings



# PARSING
def parse_skills(text: str) -> List[str]:
    lines = re.split(r"[\n,•\-]+", text)
    return [line.strip() for line in lines if line.strip()]

# STATE
class AgentState(TypedDict):
    resume: str
    jd: str
    resume_skills: List[str]
    jd_skills: List[str]
    matched: List[str]
    missing: List[str]
    score: int
#GRAPH NODES 
def extract_resume_node(state: AgentState) -> AgentState:
    skills = extract_resume_skills.invoke(state["resume"])
    return {**state, "resume_skills": skills}

def extract_jd_node(state: AgentState) -> AgentState:
    skills = extract_jd_skills.invoke(state["jd"])
    return {**state, "jd_skills": skills}

def compare_skills_node(state: AgentState) -> AgentState:
    result = compare_skills.invoke({
        "resume_skills": state["resume_skills"],
        "jd_skills": state["jd_skills"]
    })
    return {**state, "matched": result["matched"], "missing": result["missing"]}

def score_node(state: AgentState) -> AgentState:
    score = score_match.invoke({
        "matched": state["matched"],
        "jd_skills": state["jd_skills"]
    })
    return {**state, "score": score}

def youtube_node(state: AgentState) -> AgentState:
    if not state["missing"]:
        return {**state, "youtube_links": {"message": "🎉 You have all the required skills!"}}
    yt_links = get_youtube_recommendations.invoke(state["missing"])
    return {**state, "youtube_links": yt_links}
  
#AGENTIC FLOW

graph = StateGraph(AgentState)
graph.add_node("extract_resume", extract_resume_node)
graph.add_node("extract_jd", extract_jd_node)
graph.add_node("compare", compare_skills_node)
graph.add_node("score", score_node)


# Edges
graph.set_entry_point("extract_resume")
graph.add_edge("extract_resume", "extract_jd")
graph.add_edge("extract_jd", "compare")
graph.add_edge("compare", "score")

graph.add_edge("score", END)

resume_match_agent = graph.compile()

if __name__ == "__main__":
    resume = "Experienced in Python, SQL, Deep Learning, Communication, Teamwork"
    jd = "Looking for a data scientist with skills in Python, Pandas, Deep Learning, PyTorch, Communication"
    result = resume_match_agent.invoke({
        "resume": resume,
        "jd": jd
    })
    print("✅ Score:", result["score"])
    print("✅ Missing Skills:", result["missing"])
    

      
