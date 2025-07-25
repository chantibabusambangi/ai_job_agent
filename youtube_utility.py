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
        f"Suggest 5 to 7 of the best YouTube channels or video series to learn the following skills: "
        f"{skills_str}. Only return the channel/video title and link. Avoid unnecessary text."
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
        youtube_links = content.splitlines()  # Break into list
        return {
            **state,
            "youtube_links": youtube_links
        }
    else:
        return {
            **state,
            "youtube_links": [f"❌ Error {response.status_code}: {response.text}"]
        }



# Example usage
if __name__ == "__main__":
    skills_input = input("Enter missing skills (comma-separated): ").strip()
    skills = [s.strip() for s in skills_input.split(",") if s.strip()]

    sample_state = {
        "missing_skills": skills
    }

    result = youtube_utility(sample_state)
    print("\n📺 YouTube Suggestions:\n")
    for link in result.get("youtube_links", []):
        print(link)
