import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Initialize the Groq LLaMA3 model
llm = ChatGroq(temperature=0.3, model_name="llama3-8b-8192")

# Streamlit UI
st.set_page_config(page_title="Resume Score Agent", layout="centered")
st.title("🧠 Resume Scoring Agent")
st.markdown("Built by **ChantiBabuSambangi**")

jd_input = st.text_area("📄 Paste the Job Description (JD)", height=200)
resume_input = st.text_area("🧾 Paste your Resume (text only)", height=300)

if st.button("🔍 Evaluate Resume"):
    if not jd_input or not resume_input:
        st.warning("Please provide both the Job Description and Resume.")
        st.stop()

    # Define the agent
    resume_agent = Agent(
        role="Resume Scoring Agent",
        goal="Give a detailed resume score and list missing or weakly represented skills",
        backstory=(
            "You are an expert career assistant specializing in resume analysis. "
            "Your job is to compare a candidate's resume with a job description and find skills or qualifications "
            "that are either missing or weakly represented. You then generate a resume score and list those missing/weak skills."
        ),
        verbose=True,
        llm=llm
    )

    # Define the task prompt template
    prompt_template = PromptTemplate.from_template("""
    Job Description:
    {jd}

    Resume:
    {resume}

    You are to:
    1. Extract all relevant skills, technologies, and qualifications mentioned in the Job Description.
    2. Analyze the resume and identify:
        - Strongly present skills ✅
        - Weakly implied skills ⚠️
        - Missing skills ❌
    3. Return the following:
        - Resume Score (0-100): based on skill overlap and match
        - List of ❌ Missing skills
        - List of ⚠️ Weakly represented skills
        - Suggestion to improve the resume

    Format the response like:

    --- Resume Score ---
    Score: XX / 100

    --- Missing Skills (❌) ---
    - Skill A
    - Skill B

    --- Weakly Represented Skills (⚠️) ---
    - Skill C
    - Skill D

    --- Suggestions ---
    Add details or projects related to the missing or weakly represented skills.
    """)

    # Fill the task
    task_prompt = prompt_template.format(jd=jd_input, resume=resume_input)

    resume_task = Task(
        description=task_prompt,
        agent=resume_agent
    )

    # Run the crew
    try:
        crew = Crew(
            agents=[resume_agent],
            tasks=[resume_task],
            process=Process.sequential,
            verbose=True
        )
        result = crew.run()
        st.success("✅ Resume Evaluation Complete!")
        st.markdown(result)

    except Exception as e:
        st.error(f"❌ Error while running agentic workflow: {str(e)}")
