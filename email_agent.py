import re
import os
import uuid
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
import streamlit as st  # Optional, for error display in Streamlit apps

# Load environment variables
load_dotenv()

# ✅ Initialize Groq LLM (agent-specific)
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="deepseek-r1-distill-llama-70b")

# ------------------------ Extracting Info ------------------------

def extract_candidate_name(text):
    name_line = next((line for line in text.split('\n') if 'name' in line.lower()), None)
    if name_line:
        name_match = re.search(r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)', name_line)
        if name_match:
            return name_match.group(1).strip()
    return "Candidate"

def extract_skills(text):
    skills_keywords = [
        "Python", "Machine Learning", "Deep Learning", "SQL", "NLP", "Computer Vision",
        "Data Analysis", "Java", "C++", "HTML", "CSS", "JavaScript"
    ]
    return [skill for skill in skills_keywords if skill.lower() in text.lower()]

# ------------------------ PDF + LLM Utilities ------------------------

def generate_cover_letter(resume_text, jd_text):
    prompt = f"""
You are a professional resume and cover letter expert. Based on the following:

Resume:
{resume_text}

Job Description:
{jd_text}

Write a concise, professional, and personalized cover letter that aligns the candidate's skills with the job.
dont give an unnessary text and words just give the coverletter.
"""
    return llm.invoke([HumanMessage(content=prompt)]).content

def generate_qa_guide(resume_text, jd_text):
    prompt = f"""
You are an expert technical recruiter.

Given this Resume:
{resume_text}

And Job Description:
{jd_text}

Generate 10-12 likely interview questions along with model answers that the candidate can prepare.
"""
    return llm.invoke([HumanMessage(content=prompt)]).content

def save_text_to_pdf(text, filename):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    y = height - 40
    for line in text.split('\n'):
        if y < 40:
            c.showPage()
            y = height - 40
        c.drawString(40, y, line[:110])
        y -= 14
    c.save()

# ------------------------ Email Utilities ------------------------

def send_email_with_attachments(to_email, subject, body, attachments):
    from_email = os.getenv("EMAIL_USER")
    from_password = os.getenv("EMAIL_PASS")

    if not from_email or not from_password:
        raise ValueError("Email credentials are missing in environment variables.")

    message = MIMEMultipart()
    message['From'] = from_email
    message['To'] = to_email
    message['Subject'] = subject
    message.attach(MIMEText(body, 'plain'))

    for filepath in attachments:
        with open(filepath, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f"attachment; filename= {os.path.basename(filepath)}",
            )
            message.attach(part)

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(from_email, from_password)
        server.send_message(message)

# ------------------------ Main Agent ------------------------


def email_agent(resume_text, jd_text, user_email):
    candidate_name = extract_candidate_name(resume_text)
    candidate_skills = extract_skills(resume_text)

    try:
        cover_letter = generate_cover_letter(resume_text, jd_text)
    except Exception as e:
        st.error(f"[ERROR] Cover Letter Generation Failed: {e}")
        cover_letter = "Unable to generate cover letter at this time."

    try:
        qa_guide = generate_qa_guide(resume_text, jd_text)
    except Exception as e:
        st.error(f"[ERROR] Q&A Generation Failed: {e}")
        qa_guide = "Unable to generate Q&A at this time."

    # Use unique filenames to avoid collision
    cl_file = f"cover_letter_{uuid.uuid4().hex[:6]}.pdf"
    qa_file = f"qa_guide_{uuid.uuid4().hex[:6]}.pdf"

    save_text_to_pdf(cover_letter, cl_file)
    save_text_to_pdf(qa_guide, qa_file)

    subject = "📄 Your Personalized Cover Letter & Interview Guide"
    body = f"Hi {candidate_name},\n\nAttached are your AI-generated cover letter and interview Q&A guide.\n\nGood luck with your application!\n\nRegards,\nAI Job Agent"

    try:
        send_email_with_attachments(user_email, subject, body, [cl_file, qa_file])
        st.success(f"✅ Email sent to {user_email} with the generated documents.")
    except Exception as e:
        st.error(f"❌ Failed to send email: {e}")
def email_agent_node(state):
    resume_text = state["resume_text"]
    jd_text = state["jd_text"]
    user_email = state["user_email"]
    email_agent(resume_text, jd_text, user_email)
    return state  # return unchanged or modified state as needed

