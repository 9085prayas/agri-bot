from langchain_community.tools import WikipediaQueryRun, TavilySearchResults
from langchain_community.utilities import WikipediaAPIWrapper

def load_tools():
    """
    Loads the tools for the conversational agent.
    - Wikipedia: For general knowledge questions.
    - Tavily Search: For real-time, up-to-date information.
    """
    # Initialize Wikipedia tool
    # It will search Wikipedia and return a concise summary of the top result.
    wiki = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=2000)
    )

    # Initialize Tavily Search tool
    # This requires a TAVILY_API_KEY in your .env file.
    # It's a powerful search engine designed for AI agents.
    tavily_search = TavilySearchResults(max_results=3)

    return [wiki, tavily_search]
