# In settings.py

import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

class Settings:
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")
    # Using a faster, more efficient model to respect free tier limits
    MODEL: str = "gemini-2.5-flash-latest"
    TEMPERATURE: float = 0.2

settings = Settings()