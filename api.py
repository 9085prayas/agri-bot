import asyncio
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

from core.translation import translate_to_english, translate_back
from agent.rag_agent import build_rag_chain
from agent.conversational import get_conversational_agent
from core.llm import load_llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

class ChatRequest(BaseModel):
    query: str
    session_id: str = "default_session"

chat_histories: Dict[str, List[Dict[str, str]]] = {}

app = FastAPI(
    title="Agri-Bot API",
    description="An API for the Agri-Bot assistant.",
    version="1.0.0",
)

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
    
    # --- THIS IS THE FIX ---
    # The classifier now recognizes a 'Conversational' category.
    llm = load_llm()
    prompt = PromptTemplate.from_template(
        "Your task is to classify a user's input into one of four categories: 'Agricultural', 'Greeting', 'Conversational', or 'Off-topic'.\n"
        "Greetings include 'hello', 'hi', 'thank you'.\n"
        "Conversational inputs are short, simple responses like 'ok', 'yes', 'no', 'got it'.\n"
        "If the input is a follow-up to an agricultural topic, classify it as 'Agricultural'.\n\n"
        "Chat History:\n{chat_history}\n\n"
        "User Input: {user_input}\n\n"
        "Classification:"
    )
    models["classifier_chain"] = prompt | llm | StrOutputParser()
    print("--- Models loaded successfully. API is ready. ---")

def clean_and_split_for_ui(text: str) -> List[str]:
    """
    Strips markdown and splits the text into a list of lines for easy UI rendering.
    """
    text = text.replace('**', '')
    text = re.sub(r'^\* ', '- ', text, flags=re.MULTILINE)
    text = re.sub(r'^### ', '', text, flags=re.MULTILINE)
    text = re.sub(r'^## ', '', text, flags=re.MULTILINE)
    text = re.sub(r'^# ', '', text, flags=re.MULTILINE)
    lines = text.strip().split('\n')
    return [line.strip() for line in lines if line.strip()]

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
        classification = await classifier.ainvoke({
            "user_input": query,
            "chat_history": langchain_chat_history
        })

        # --- THIS IS THE FIX ---
        # Added logic to handle the new 'Conversational' category.
        if "greeting" in classification.lower():
            if any(word in query.lower() for word in ['thank', 'thanks']):
                final_response = "You're welcome! Is there anything else I can help you with regarding agriculture?"
            else:
                final_response = "Hello! I am Agri-Advisor. How can I assist you with your farming questions today?"
        elif "conversational" in classification.lower():
            final_response = "Is there anything else I can help you with?"
        elif "off-topic" in classification.lower():
            final_response = "I am Agri-Bot, your farming assistant. I can only answer questions related to agriculture."
        else:
            # Handle agricultural questions
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
        # ---------------------

        cleaned_response_lines = clean_and_split_for_ui(final_response)
        full_response_string = "\n".join(cleaned_response_lines)

        chat_histories[session_id].append({"type": "human", "content": query})
        chat_histories[session_id].append({"type": "ai", "content": full_response_string})

        return {"response": full_response_string, "session_id": session_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
