import streamlit as st
from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import PyPDFLoader

from resume_score_agent import resume_skill_match_agent
from email_agent import email_agent_node as email_agent
from youtube_utility import youtube_utility

import tempfile

# ------------- IMPLEMENT YOUR LLM CALL HERE --------------
def call_llm_router(prompt: str) -> str:
    """
    Connect this function to your LLM provider (Groq, OpenAI, etc.)
    Send the prompt and parse the single keyword response indicating next action.
    Return one of: "resume_skill_match", "youtube", "email", "end"

    For now, here is a simple placeholder that returns a fixed step order.
    Replace with actual LLM call code!
    """
    # Example static cycling through steps for testing
    # In production, replace this with your LLM call and parsing logic
    call_llm_router.counter = getattr(call_llm_router, "counter", 0) + 1
    steps = ["resume_skill_match", "youtube", "email", "end"]
    return steps[call_llm_router.counter - 1]

# ------------- LLM Router Node --------------
def llm_router(state: dict) -> dict:
    """
    Uses LLM to decide which agent to run next given current state.
    """
    prompt = f"""
You are a multi-agent controller deciding which tool to run next.
Current state info:
- Has resume been scored? {"Yes" if state.get('score') is not None else "No"}
- Missing skills: {state.get('missing_skills', [])}
- User provided email? {"Yes" if state.get('user_email') else "No"}

Choose one action from: resume_skill_match, youtube, email, end
Respond exactly with one of these words (no explanation):

"""
    action = call_llm_router(prompt).strip().lower()
    valid_actions = {"resume_skill_match", "youtube", "email", "end"}
    if action not in valid_actions:
        action = "end"
    return {"action": action}

# ------------- Build Graph --------------
builder = StateGraph(dict)

builder.add_node("router", llm_router)
builder.add_node("resume_skill_match", resume_skill_match_agent)
builder.add_node("youtube", youtube_utility)
builder.add_node("email", email_agent)

builder.set_entry_point("router")

graph = builder.compile()

# ------------- Utilities --------------
def convert_to_text(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        return PyPDFLoader(tmp_path).load()[0].page_content
    else:
        return uploaded_file.read().decode("utf-8")

# ------------- Agentic Invocation Loop --------------
def agentic_invoke(graph, initial_state):
    state = initial_state.copy()

    while True:
        # Ask router for next action
        router_output = graph.invoke({"name": "router"}, state)
        action = router_output.get("action")

        if action == "end" or action is None:
            break

        # Call selected agent node
        action_output = graph.invoke({"name": action}, state)
        # Update state with outputs received
        state.update(action_output)

    return state

# ------------- Streamlit UI --------------
st.set_page_config(page_title="AI Job Agent System", layout="centered")
st.title("🚀 AI Job Agent System")

uploaded_resume = st.file_uploader("📄 Upload Resume (PDF only)", type=["pdf"])
uploaded_jd = st.file_uploader("📄 Upload Job Description (PDF or .txt)", type=["pdf", "txt"])
user_email = st.text_input("📧 Enter your Email")

button_disabled = not (uploaded_resume and uploaded_jd and user_email)
st.sidebar.markdown("🔹 **Built with ❤️ by chantibabusambangi@gmail.com**")
if st.button("🚀 Run AI Agent Pipeline", disabled=button_disabled):
    st.info("Running agentic AI workflow...")

    resume_text = convert_to_text(uploaded_resume)
    jd_text = convert_to_text(uploaded_jd)

    from email_agent import extract_skills
    job_skills = extract_skills(jd_text)

    initial_state = {
        "resume_text": resume_text,
        "jd_text": jd_text,
        "job_skills": job_skills,
        "user_email": user_email
    }

    try:
        output = agentic_invoke(graph, initial_state)

        st.subheader("✅ Final Agent Output:")

        if "score" in output:
            st.metric("📊 Resume Match Score", f"{output['score']:.2f}%")

        if "missing_skills" in output and output["missing_skills"]:
            st.markdown("🧠 **Missing Skills:**")
            st.write(", ".join(output["missing_skills"]))
        else:
            st.markdown("🧠 **No missing skills detected!**")

        if "youtube_links" in output:
            st.markdown("🎥 **YouTube Suggestions:**")
            for suggestion in output["youtube_links"]:
                st.markdown(f"- {suggestion}")

    except Exception as e:
        st.error(f"❌ Error while running agentic workflow: {e}")
