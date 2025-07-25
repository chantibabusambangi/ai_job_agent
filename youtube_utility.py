#youtube video suggestions for upskill the missing skills
import os
import requests
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama3-70b-8192"

headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

def suggest_youtube_channels(missing_skills):
    if not missing_skills or not isinstance(missing_skills, list):
        return "🎉 Congratulations! You have all the required skills for this job."


    # Create a comma-separated string of skills
    skills_str = ', '.join(missing_skills)

    # Prompt to the LLM
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

    # Call the Groq API
    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        content = response.json()['choices'][0]['message']['content'].strip()
        return content
    else:
        return f"❌ Error {response.status_code}: {response.text}"

# Example usage
if __name__ == "__main__":
    # Sample input
    skills = ["Redis", "Docker", "Kafka"]
    result = suggest_youtube_channels(skills)
    print("\n📺 YouTube Suggestions:\n")
    print(result)
