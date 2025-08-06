# youtube video suggestions for upskill the missing skills
import os
import requests
from dotenv import load_dotenv
import re

# Load API key from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.1-8b-instant"

headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

def youtube_utility(state):
    missing_skills = state.get("missing_skills", [])
    
    if not missing_skills or not isinstance(missing_skills, list):
        return {
            **state,
            "youtube_links": ["🎉 Congratulations! You have all the required skills for this job."]
        }

    skills_str = ', '.join(missing_skills)

    prompt = (
        f"You are an AI assistant. For each of the following skills, suggest 1 to 2 high-quality YouTube videos to help someone upskill: {skills_str}.\n\n"
        "Format the output exactly like this for each skill:\n"
        "Skill: SkillName\n"
        "- [Video Title](https://youtube.com/...) - Channel: Channel Name\n"
        "- [Another Title](https://youtube.com/...) - Channel: Channel Name\n\n"
        "Only return the list in Markdown format. No explanations or extra text."
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that suggests YouTube learning resources."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=25)

    if response.status_code == 200:
        content = response.json()['choices'][0]['message']['content'].strip()
        lines = content.splitlines()

        youtube_links = []
        current_skill = ""

        for line in lines:
            skill_header = re.match(r'^Skill:\s*(.+)', line)
            video_match = re.search(r'\[(.*?)\]\((https?://.*?)\)\s*-\s*Channel:\s*(.+)', line)

            if skill_header:
                current_skill = f"**🧠 Skill: {skill_header.group(1).strip()}**"
                youtube_links.append(current_skill)
            elif video_match:
                title = video_match.group(1).strip()
                url = video_match.group(2).strip()
                channel = video_match.group(3).strip()
                youtube_links.append(f"- [{title}]({url}) — 🎥 Channel: **{channel}**")
            elif line.strip():
                youtube_links.append(line.strip())

        return {
            **state,
            "youtube_links": youtube_links
        }

    else:
        return {
            **state,
            "youtube_links": [f"❌ Error fetching suggestions: {response.status_code}"]
        }
