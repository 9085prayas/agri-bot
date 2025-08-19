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
    
    llm = load_llm()
    
    # Classifier for user intent
    classifier_prompt = PromptTemplate.from_template(
        "Your task is to classify a user's input into one of the following categories: 'Agricultural', 'Greeting', 'Conversational', 'Showing Gratitude', 'Being Polite / Making Requests', 'Farewells', 'Capability_Inquiry', or 'Off-topic'.\n"
        "Showing Gratitude includes phrases like 'thank you', 'thanks', 'I appreciate it'.\n"
        "Being Polite / Making Requests includes phrases like 'could you please', 'would you mind', 'I would like to request'.\n"
        "Farewells include 'goodbye', 'see you later', 'take care'.\n"
        "Greetings include 'hello', 'hi','Good morning','Good afternoon','Good evening'.\n"
        "Conversational inputs are short, simple responses like 'ok', 'yes', 'no', 'got it','Alright','Of course','Definitely','Nah','No way','Not really','I don't think so','Maybe','Perhaps','Possibly','I'm not sure','We'll see','Got it','I see','Right','True','Nothing' .\n"
        "Capability_Inquiry includes questions about what you can do, like 'what can you do?','tell me about yourself','List your features','What kind of things can you help me with?','What is your purpose?'.\n"
        "If the input is a follow-up to an agricultural topic, classify it as 'Agricultural'.\n\n"
        "Chat History:\n{chat_history}\n\n"
        "User Input: {user_input}\n\n"
        "Classification:"
    )
    models["classifier_chain"] = classifier_prompt | llm | StrOutputParser()

    # --- THIS IS THE FIX ---
    # New, more detailed prompt for generating high-quality suggestions.
    suggestion_prompt = PromptTemplate.from_template(
        "You are an expert AI assistant for an agricultural bot named 'Agri-Advisor'. Your task is to generate 3 highly relevant and insightful follow-up questions based on a user's query and the bot's response. These suggestions should anticipate the user's next logical thought and guide them towards deeper knowledge.\n\n"
        "**Rules for Generating Suggestions:**\n"
        "1.  **Be Proactive:** Think like an expert advisor. What would a farmer or agricultural professional ask next?\n"
        "2.  **Be Specific:** Avoid generic questions. The suggestions should be directly related to the topics, crops, or schemes mentioned in the conversation.\n"
        "3.  **Cover Different Angles:** Try to provide suggestions that explore different facets of the topic, such as:\n"
        "    - **Practical Application:** How can the user apply this information? (e.g., 'What is the step-by-step process to apply for this scheme?')\n"
        "    - **Financial Implications:** What are the costs or benefits? (e.g., 'What is the estimated cost of this fertilizer per acre?')\n"
        "    - **Deeper Dive:** Ask for more detail on a sub-topic. (e.g., 'Tell me more about the specific pests that affect the Swarna rice variety.')\n"
        "4.  **Format:** Return the questions as a single, comma-separated string. Do not include numbers, bullet points, or any other formatting.\n\n"
        "**Example:**\n"
        "User Query: 'What is the PM-KISAN scheme?'\n"
        "Bot Response: 'The PM-KISAN scheme is a government initiative that provides income support of ₹6,000 per year to eligible farmer families.'\n"
        "Suggestions: What are the eligibility criteria for PM-KISAN?, How can I apply for the PM-KISAN scheme?, When is the next installment paid?\n\n"
        "**Current Conversation:**\n"
        "User Query: {query}\n"
        "Bot Response: {response}\n\n"
        "**Generated Suggestions (comma-separated list):**"
    )
    models["suggestion_chain"] = suggestion_prompt | llm | StrOutputParser()
    # ---------------------

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

        classification_lower = classification.lower()
        suggestions = []
        
        if "greeting" in classification_lower:
            final_response = "Hello! I am Agri-Advisor. How can I assist you with your farming questions today?"
        elif "showing gratitude" in classification_lower:
            final_response = "You're welcome! Is there anything else I can help you with regarding agriculture?"
        elif "farewells" in classification_lower:
            final_response = "Goodbye! Feel free to reach out if you have more agricultural questions."
        elif "conversational" in classification_lower or "being polite" in classification_lower:
             final_response = "Of course. What would you like to know about agriculture?"
        elif "capability_inquiry" in classification_lower:
            translated_query, original_lang = translate_to_english(query)
            agent = models["agent"]
            agent_response = await agent.ainvoke({
                "input": translated_query,
                "chat_history": langchain_chat_history
            })
            final_response = agent_response.get("output", "Sorry, I could not find an answer.")
            final_response = translate_back(final_response, original_lang)
        elif "off-topic" in classification_lower:
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
            
            # Generate suggestions only for valid agricultural responses.
            suggestion_chain = models["suggestion_chain"]
            suggestion_text = await suggestion_chain.ainvoke({"query": query, "response": final_response})
            suggestions = [s.strip() for s in suggestion_text.split(',') if s.strip()]

        cleaned_response_lines = clean_and_split_for_ui(final_response)
        full_response_string = "\n".join(cleaned_response_lines)

        chat_histories[session_id].append({"type": "human", "content": query})
        chat_histories[session_id].append({"type": "ai", "content": full_response_string})

        return {"response": full_response_string, "session_id": session_id, "suggestions": suggestions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
