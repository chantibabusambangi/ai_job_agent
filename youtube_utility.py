#youtube video suggestions for upskill the missing skills
import os
import requests
from dotenv import load_dotenv

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
        f"You are an AI assistant! For each of the following skills, suggest 1 to 2 high-quality YouTube videos or channels to learn them: "
        f"{skills_str}. Only return the video/channel title and clickable YouTube link in Markdown format."
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
        for line in lines:
            match = re.search(r'(.+?)\s*-\s*(https?://[^\s]+)', line)
            if match:
                title = match.group(1).strip()
                url = match.group(2).strip()
                youtube_links.append(f"[{title}]({url})")
            else:
                youtube_links.append(line.strip())
    
        return {
            **state,
            "youtube_links": youtube_links
        }
    
    
