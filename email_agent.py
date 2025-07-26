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
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile")

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
        You are an expert resume writer.
        Generate a concise and formal **cover letter** for the candidate using the following inputs:
        ---
        **Resume:**
        {resume_text}
        ---
        **Job Description:**
        {jd_text}
        ---
Instructions:
- Start with the candidate’s name, city, phone number, and email at the top.
- make sure to Render each section exactly as labeled (HEADER, RECIPIENT, BODY).
- In BODY, break into 4 distinct paragraphs.
- No extra commentary or headings—just the formatted letter.
- Include today’s date.
- Add “Hiring Manager” and company name.
- Use 3–4 short paragraphs in a professional tone:
    1. Introduce the candidate and express interest
    2. Highlight experience and align it with the job
    3. Share enthusiasm and appreciation
- Avoid unnecessary filler or generic phrases.
- Return only the clean, formatted letter — no explanations or extra notes.
        """
    return llm.invoke([HumanMessage(content=prompt)]).content
def generate_qa_guide(resume_text, jd_text):
    prompt = f"""
You are a technical recruiter and career coach.

Step 1️⃣ Identify the top 3–4 technical domains or skill‑clusters mentioned in the resume or job description.

Step 2️⃣ For each domain, write 2–3 focused Q&A pairs numbered sequentially.

Step 3️⃣ Add behavioral or situational questions so that there are 10–12 questions total.

---
**Resume Text:**
{resume_text}

**Job Description Text:**
{jd_text}

**Instructions:**
- Use this exact numbering format:
  Q1: [first question]
  A1: [first answer]
  Q2: [second question]
  A2: [second answer]
  …and so on.
- Ensure each answer is 2–4 lines and clearly tailored to the candidate’s experience.
- Return only the numbered Q&A pairs, no extra headings or commentary.
"""
    return llm.invoke([HumanMessage(content=prompt)]).content

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch

def save_text_to_pdf(text, filename):
    """
    Writes `text` into a nicely formatted PDF at `filename`.
    Uses ReportLab Platypus to auto-wrap paragraphs, handle page breaks, and set margins.
    """
    # Set up document with 1" margins
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=inch,
        rightMargin=inch,
        topMargin=inch,
        bottomMargin=inch
    )
    
    # Create a simple paragraph style
    styles = getSampleStyleSheet()
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=12,
        leading=16,         # line spacing
        spaceAfter=12       # space after each paragraph
    )
    
    # Split on double-newlines for paragraphs
    story = []
    for para in text.strip().split('\n\n'):
        # Convert any single-line breaks into spaces
        clean = ' '.join(line.strip() for line in para.splitlines())
        story.append(Paragraph(clean, body_style))
        story.append(Spacer(1, 0.2*inch))
    
    doc.build(story)

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

def clean_text_for_pdf(text):
    lines = text.strip().split('\n')
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    return '\n'.join(cleaned_lines)
def email_agent(resume_text, jd_text, user_email):
    candidate_name = extract_candidate_name(resume_text)
    candidate_skills = extract_skills(resume_text)

    try:
        cover_letter = generate_cover_letter(resume_text, jd_text)
    except Exception as e:
        print(f"[ERROR] Cover Letter Generation Failed: {e}")
        cover_letter = "Unable to generate cover letter at this time."

    try:
        qa_guide = generate_qa_guide(resume_text, jd_text)
    except Exception as e:
        print(f"[ERROR] Q&A Generation Failed: {e}")
        qa_guide = "Unable to generate Q&A at this time."

    cl_file = f"cover_letter_{uuid.uuid4().hex[:6]}.pdf"
    qa_file = f"qa_guide_{uuid.uuid4().hex[:6]}.pdf"

    clean_cl = clean_text_for_pdf(cover_letter)
    clean_qa = clean_text_for_pdf(qa_guide)

    save_text_to_pdf(clean_cl, cl_file)
    save_text_to_pdf(clean_qa, qa_file)

    subject = "📄 Your Personalized Cover Letter & Interview Guide"
    body = f"Hi {candidate_name},\n\nAttached are your AI-generated cover letter and interview Q&A guide.\n\nGood luck with your application!\n\nRegards,\nAI Job Agent"

    try:
        send_email_with_attachments(user_email, subject, body, [cl_file, qa_file])
        print(f"[SUCCESS] Email sent to {user_email} with generated documents.")
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")
def email_agent_node(state):
    resume_text = state["resume_text"]
    jd_text = state["jd_text"]
    user_email = state["user_email"]
    email_agent(resume_text, jd_text, user_email)
    return state  # return unchanged or modified state as needed

