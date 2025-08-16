import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
    BASE_URL: str = "https://api.groq.com/openai/v1"
    MODEL: str = "llama3-70b-8192"
    TEMPERATURE: float = 0.2

settings = Settings()