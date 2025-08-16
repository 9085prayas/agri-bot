from langchain.agents import initialize_agent, AgentType
from core.llm import load_llm
from core.tools import load_tools
from core.memory import get_memory

def get_conversational_agent():
    llm = load_llm()
    tools = load_tools()
    memory = get_memory()

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        max_iterations=8,
        handle_parsing_errors=True,
    )
