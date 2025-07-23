import streamlit as st
from langgraph.graph import StateGraph, END
from agents import resume_score_agent, cover_letter_agent, qa_generator_agent, mail_sender_agent
from router import router

# Define the shared graph state
class GraphState(dict): pass

# Create LangGraph
builder = StateGraph(GraphState)

# Nodes
builder.add_node("router", router)
builder.add_node("resume_score", resume_score_agent)
builder.add_node("cover_letter", cover_letter_agent)
builder.add_node("qa_generator", qa_generator_agent)
builder.add_node("mail_sender", mail_sender_agent)

# Edges
builder.set_entry_point("router")
builder.add_edge("router", "resume_score")
builder.add_edge("router", "cover_letter")
builder.add_edge("router", "qa_generator")
builder.add_edge("router", "mail_sender")
builder.add_edge("resume_score", END)
builder.add_edge("cover_letter", END)
builder.add_edge("qa_generator", END)
builder.add_edge("mail_sender", END)

graph = builder.compile()

# Streamlit UI
st.set_page_config(page_title="AI Job Assist", layout="centered")
st.title("🧠 AI Job Agent System")

uploaded_resume = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
uploaded_jd = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])
user_email = st.text_input("Enter your Email")

selected_task = st.selectbox("Select Task", ["Resume Score", "Cover Letter", "Q&A Generator", "Mail Sender"])

if st.button("Run Agent"):
    if not uploaded_resume:
        st.warning("Please upload your resume.")
    elif selected_task in ["Resume Score", "Q&A Generator"] and not uploaded_jd:
        st.warning("Please upload job description.")
    elif selected_task == "Mail Sender" and not user_email:
        st.warning("Email is required for Mail Agent.")
    else:
        st.success(f"Running `{selected_task}` Agent...")

        # Prepare graph state
        state = {
            "resume_pdf": uploaded_resume,
            "jd_pdf": uploaded_jd,
            "user_email": user_email,
            "task": selected_task.lower().replace(" ", "_")
        }

        output = graph.invoke(state)
        st.subheader("Agent Output:")
        st.write(output["result"])
