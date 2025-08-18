from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import settings

def load_llm():
    """Load the Google Gemini LLM"""
    # The deprecated 'convert_system_message_to_human' parameter has been removed.
    return ChatGoogleGenerativeAI(
        model=settings.MODEL,
        temperature=settings.TEMPERATURE,
        google_api_key=settings.GOOGLE_API_KEY,
    )
