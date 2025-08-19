import streamlit as st
import asyncio
from core.translation import translate_to_english, translate_back
from core.memory import get_memory
from agent.rag_agent import build_rag_chain
from agent.conversational import get_conversational_agent
from core.llm import load_llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- FIX FOR ASYNCIO EVENT LOOP ERROR ---
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
# -----------------------------------------

# --- LAZY INITIALIZATION WITH CACHING ---
@st.cache_resource
def get_rag_chain():
    """Builds and returns the RAG chain."""
    return build_rag_chain()

@st.cache_resource
def get_agent():
    """Builds and returns the conversational agent with tools."""
    return get_conversational_agent()

@st.cache_resource
def get_classifier_chain():
    """Builds a chain to classify user queries."""
    llm = load_llm()
    # --- THIS IS THE FIX ---
    # The classifier now recognizes a 'Conversational' category for simple words.
    prompt = PromptTemplate.from_template(
        "Your task is to classify a user's input into one of four categories: 'Agricultural', 'Greeting', 'Conversational', 'Showing Gratitude' , 'Being Polite / Making Requests' , 'Farewells' or 'Off-topic'.\n"
        "Showing Gratitude includes phrases like 'thank you', 'thanks', 'I appreciate it'.\n"
        "Being Polite / Making Requests includes phrases like 'could you please', 'would you mind', 'I would like to request'.\n"
        "Farewells include 'goodbye', 'see you later', 'take care'.\n"
        "Greetings include 'hello', 'hi','Good morning','Good afternoon','Good evening'.\n"
        "Conversational inputs are short, simple responses like 'ok', 'yes', 'no', 'got it','Alright','Of course','Definitely','Nah','No way','Not really','I don't think so','Maybe','Perhaps','Possibly','I'm not sure','We'll see','Got it','I see','Right','True' .\n"
        "If the input is a follow-up to an agricultural topic, classify it as 'Agricultural'.\n\n"
        "Chat History:\n{chat_history}\n\n"
        "User Input: {user_input}\n\n"
        "Classification:"
    )
    return prompt | llm | StrOutputParser()

# --- RESILIENT STARTUP LOGIC ---
if 'rag_enabled' not in st.session_state:
    try:
        get_rag_chain()
        st.session_state.rag_enabled = True
    except Exception as e:
        st.session_state.rag_enabled = False
        print(f"--- RAG Chain initialization failed: {e}. Bot will use agent-only mode. ---")

def render_chat_ui():
    """Renders the main chat interface for the Streamlit app."""
    st.title("🌱 Agri-Bot")
    st.subheader("Your Farming Assistant")

    if "memory" not in st.session_state:
        st.session_state.memory = get_memory()
    memory = st.session_state.memory

    if st.button("New Conversation"):
        memory.clear()
        st.success("Chat history cleared!")
        st.rerun()

    for msg in memory.messages:
        st.chat_message(msg.type).markdown(msg.content)

    prompt = st.chat_input("Ask me anything about agriculture...")
    if prompt:
        with st.spinner("Thinking..."):
            try:
                chat_history = memory.messages
                classifier_chain = get_classifier_chain()
                classification = classifier_chain.invoke({
                    "user_input": prompt,
                    "chat_history": chat_history
                })

                # --- THIS IS THE FIX ---
                # Added logic to handle the new 'Conversational' category.
                if "greeting" in classification.lower():
                    if any(word in prompt.lower() for word in ['thank', 'thanks']):
                        final_translated_response = "You're welcome! Is there anything else I can help you with regarding agriculture?"
                    else:
                        final_translated_response = "Hello! I am Agri-Advisor. How can I assist you with your farming questions today?"
                elif "conversational" in classification.lower():
                    final_translated_response = "Is there anything else I can help you with?"
                elif "off-topic" in classification.lower():
                    final_translated_response = "I am Agri-Bot, your farming assistant. I can only answer questions related to agriculture."
                else:
                    # Handle agricultural questions
                    translated_query, original_lang = translate_to_english(prompt)
                    
                    if st.session_state.rag_enabled:
                        rag_chain = get_rag_chain()
                        rag_response_data = rag_chain.invoke({
                            "input": translated_query,
                            "chat_history": chat_history
                        })
                        rag_response = rag_response_data.get("answer", "").strip()

                        fallback_phrases = ["don't know", "do not have enough information", "cannot answer"]
                        if not rag_response or any(phrase in rag_response.lower() for phrase in fallback_phrases):
                            agent = get_agent()
                            agent_response = agent.invoke({
                                "input": translated_query,
                                "chat_history": chat_history
                            })
                            final_response = agent_response.get("output", "Sorry, I could not find an answer.")
                        else:
                            final_response = rag_response
                    else:
                        agent = get_agent()
                        agent_response = agent.invoke({
                            "input": translated_query,
                            "chat_history": chat_history
                        })
                        final_response = agent_response.get("output", "Sorry, I could not find an answer.")

                    final_translated_response = translate_back(final_response, original_lang)
                # ---------------------
                
                memory.add_user_message(prompt)
                memory.add_ai_message(final_translated_response)
                
                st.rerun()

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")

def main():
    """Main function to run the Streamlit app."""
    render_chat_ui()

if __name__ == "__main__":
    main()
