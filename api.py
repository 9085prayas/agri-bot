import asyncio
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

# Import your existing project's core logic
from core.translation import translate_to_english, translate_back
from agent.rag_agent import build_rag_chain
from agent.conversational import get_conversational_agent
from core.llm import load_llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# --- Data model for the API request ---
class ChatRequest(BaseModel):
    query: str
    session_id: str = "default_session"

# --- In-memory storage for conversation histories ---
chat_histories: Dict[str, List[Dict[str, str]]] = {}

# --- FastAPI Application Setup ---
app = FastAPI(
    title="Agri-Bot API",
    description="An API for the Agri-Bot assistant.",
    version="1.0.0",
)

# --- Load models on startup ---
models = {}

@app.on_event("startup")
async def startup_event():
    """On API startup, load all necessary models and chains."""
    print("--- Loading models on startup... ---")
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    models["rag_chain"] = build_rag_chain()
    models["agent"] = get_conversational_agent()
    
    llm = load_llm()
    prompt = PromptTemplate.from_template(
        "Your task is to classify if a question is related to agriculture. "
        "Answer with only the word 'Yes' or 'No'.\n\n"
        "Question: {user_input}\n\n"
        "Classification:"
    )
    models["classifier_chain"] = prompt | llm | StrOutputParser()
    print("--- Models loaded successfully. API is ready. ---")

# --- THIS IS THE FIX ---
# This function now returns a list of strings, where each string is a line.
def clean_and_split_for_ui(text: str) -> List[str]:
    """
    Strips markdown and splits the text into a list of lines for easy UI rendering.
    """
    # Remove bolding
    text = text.replace('**', '')
    # Replace markdown list items with a dash
    text = re.sub(r'^\* ', '- ', text, flags=re.MULTILINE)
    # Replace markdown headings
    text = re.sub(r'^### ', '', text, flags=re.MULTILINE)
    text = re.sub(r'^## ', '', text, flags=re.MULTILINE)
    text = re.sub(r'^# ', '', text, flags=re.MULTILINE)

    # Split the cleaned text into a list of lines and remove any empty lines
    lines = text.strip().split('\n')
    return [line.strip() for line in lines if line.strip()]
# ---------------------

# --- API Endpoint ---
@app.post("/chat", summary="Get a response from Agri-Bot")
async def chat_endpoint(request: ChatRequest):
    """Processes a user's query and returns Agri-Bot's response."""
    try:
        query = request.query
        session_id = request.session_id

        if session_id not in chat_histories:
            chat_histories[session_id] = []
        
        langchain_chat_history = [HumanMessage(content=msg["content"]) if msg["type"] == "human" else AIMessage(content=msg["content"]) for msg in chat_histories[session_id]]

        classifier = models["classifier_chain"]
        classification = await classifier.ainvoke({"user_input": query})

        if "no" in classification.lower():
            final_response = "I am Agri-Bot, your farming assistant. I can only answer questions related to agriculture."
        else:
            translated_query, original_lang = translate_to_english(query)
            
            rag_chain = models["rag_chain"]
            rag_response_data = await rag_chain.ainvoke({
                "input": translated_query,
                "chat_history": langchain_chat_history
            })
            rag_response = rag_response_data.get("answer", "").strip()

            fallback_phrases = ["don't know", "do not have enough information", "cannot answer"]
            if not rag_response or any(phrase in rag_response.lower() for phrase in fallback_phrases):
                agent = models["agent"]
                agent_response = await agent.ainvoke({
                    "input": translated_query,
                    "chat_history": langchain_chat_history
                })
                final_response = agent_response.get("output", "Sorry, I could not find an answer.")
            else:
                final_response = rag_response
            
            final_response = translate_back(final_response, original_lang)

        # Clean the response and split it into a list of lines
        cleaned_response_lines = clean_and_split_for_ui(final_response)

        # Join the lines back together for storing in history, but send the list in the response
        chat_histories[session_id].append({"type": "human", "content": query})
        chat_histories[session_id].append({"type": "ai", "content": "\n".join(cleaned_response_lines)})

        return {"response": cleaned_response_lines, "session_id": session_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
