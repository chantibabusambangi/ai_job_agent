import streamlit as st
from langgraph.graph import StateGraph, END
from langchain.document_loaders import PyPDFLoader, TextLoader
from resume_skill_match_agent import resume_skill_match_agent
from email_agent import email_agent
from youtube_utility import youtube_utility

# Define graph state class
#class GraphState(dict): pass





from langgraph.graph import StateGraph, END

builder = StateGraph(dict)

builder.add_node("resume_skill_match", resume_skill_match_agent)
builder.add_node("youtube", youtube_utility)
builder.add_node("email", email_agent)

builder.set_entry_point("resume_skill_match")

# Explicit edges showing how data flows
builder.add_edge("resume_skill_match", "youtube")
builder.add_edge("youtube", "email")
builder.add_edge("email", END)

graph = builder.compile()


# Streamlit UI
st.set_page_config(page_title="AI Job Agent System", layout="centered")
st.title("🚀 AI Job Agent System")

uploaded_resume = st.file_uploader("📄 Upload Resume (PDF only)", type=["pdf"])
uploaded_jd = st.file_uploader("📄 Upload Job Description (PDF or .txt)", type=["pdf", "txt"])
user_email = st.text_input("📧 Enter your Email")

# Convert file to text
import tempfile
from langchain.document_loaders import PyPDFLoader

def convert_to_text(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        return PyPDFLoader(tmp_path).load()[0].page_content
    else:
        return uploaded_file.read().decode("utf-8")
        
if st.button("🚀 Run AI Agent Pipeline"):
    if not uploaded_resume or not uploaded_jd or not user_email:
        st.warning("Please upload both files and enter your email.")
    else:
        st.info("Running Resume Skill Match → YouTube Suggestions → Email Agent...")
        resume_text = convert_to_text(uploaded_resume)
        jd_text = convert_to_text(uploaded_jd)

        state = {
            "resume_text": resume_text,
            "jd_text": jd_text,
            "user_email": user_email
        }

        try:
            output = graph.invoke(state)
            st.subheader("✅ Final Agent Output:")
            st.write(output.get("result", "No result returned."))  # ← Right after this
        
            # 🔽 Add this block here
            if "score" in output:
                st.metric("📊 Resume Match Score", f"{output['score']:.2f}%")
        
            if "missing_skills" in output:
                st.markdown("🧠 **Missing Skills:**")
                st.write(", ".join(output["missing_skills"]))
        
            if "youtube_links" in output:
                st.markdown("🎥 **YouTube Suggestions:**")
                for link in output["youtube_links"]:
                    st.markdown(f"- [Watch Video]({link})")
        
        except Exception as e:
            st.error(f"❌ Error: {e}")
