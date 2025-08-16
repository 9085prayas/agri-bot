from langchain_openai import ChatOpenAI
from config.settings import settings

def load_llm():
    """Load Groq LLM via OpenAI-compatible API"""
    return ChatOpenAI(
        model=settings.MODEL,
        temperature=settings.TEMPERATURE,
        api_key=settings.GROQ_API_KEY,
        base_url=settings.BASE_URL,
    )
