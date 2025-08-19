from langchain_community.tools import WikipediaQueryRun, TavilySearchResults
from langchain_community.utilities import WikipediaAPIWrapper

def load_tools():
    """
    Loads the tools for the conversational agent.
    - Tavily Search: For real-time, up-to-date information from the web.
    - Wikipedia: For general knowledge questions.
    """
    # Initialize Tavily Search tool
    # This requires a TAVILY_API_KEY in your .env file.
    tavily_search = TavilySearchResults(max_results=10)

    # Initialize Wikipedia tool
   # wiki = WikipediaQueryRun(
      #  api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=2000)
    #)

    return [tavily_search]
