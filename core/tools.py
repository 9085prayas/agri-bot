from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun, TavilySearchResults
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper

def load_tools():
    wiki = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    )
    arxiv = ArxivQueryRun(
        api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    )
    tavily_search = TavilySearchResults(max_results=3)

    return [wiki, arxiv, tavily_search]
