import re
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq  # ✅ Groq LLM added

load_dotenv()

# ✅ Initialize Groq LLM (agent-specific)
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="mixtral-8x7b-32768")

def extract_candidate_name(text):
    name_line = next((line for line in text.split('\n') if 'name' in line.lower()), None)
    if name_line:
        name_match = re.search(r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)', name_line)
        if name_match:
            return name_match.group(1).strip()
    return "Candidate"

def extract_skills(text):
    skills_keywords = ["Python", "Machine Learning", "Deep Learning", "SQL", "NLP", "Computer Vision", "Data Analysis", "Java", "C++", "HTML", "CSS", "JavaScript"]
    return [skill for skill in skills_keywords if skill.lower() in text.lower()]

# ✅ Replaced get_llm_response with Groq call
def generate_cover_letter(resume_text, jd_text):
    prompt = f"""
    Given the following resume:
    {resume_text}

    And this job description:
    {jd_text}

    Write a personalized and professional cover letter matching the job role.
    """
    return llm.invoke([HumanMessage(content=prompt)]).content

def generate_qa_guide(resume_text, jd_text):
    prompt = f"""
    Given the resume:
    {resume_text}

    And this job description:
    {jd_text}

    Generate a list of top 10 possible interview questions and answers.
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

def send_email_with_attachments(to_email, subject, body, attachments):
    from_email = os.getenv("EMAIL_USER")
    from_password = os.getenv("EMAIL_PASS")

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

def email_agent(resume_text, jd_text, user_email):
    candidate_name = extract_candidate_name(resume_text)
    candidate_skills = extract_skills(resume_text)
    
    try:
        cover_letter = generate_cover_letter(resume_text, jd_text)
    except Exception as e:
        print(f"[ERROR] Failed to generate cover letter: {e}")
        cover_letter = "Unable to generate cover letter at this time."
    
    try:
        qa_guide = generate_qa_guide(resume_text, jd_text)
    except Exception as e:
        print(f"[ERROR] Failed to generate Q&A: {e}")
        qa_guide = "Unable to generate Q&A at this time."
        
    save_text_to_pdf(cover_letter, "cover_letter.pdf")
    save_text_to_pdf(qa_guide, "qa_guide.pdf")

    subject = "Your Personalized Cover Letter & Interview Q&A Guide"
    body = f"Hi {candidate_name},\n\nPlease find attached your auto-generated cover letter and interview Q&A guide.\n\nGood luck!"
    attachments = ["cover_letter.pdf", "qa_guide.pdf"]

    send_email_with_attachments(user_email, subject, body, attachments)

    print(f"[INFO] Email sent to {user_email} with attachments.")
