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

@st.cache_data
def classify_query(user_input: str) -> str:
    """Classifies if a query is agricultural using a cached LLM call."""
    llm = load_llm()
    prompt = PromptTemplate.from_template(
        "Your task is to classify if a question is related to agriculture. "
        "Answer with only the word 'Yes' or 'No'.\n\n"
        "Question: {user_input}\n\n"
        "Classification:"
    )
    classifier_chain = prompt | llm | StrOutputParser()
    return classifier_chain.invoke({"user_input": user_input})
# ----------------------------------------------------

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

    # Display existing messages from history
    for msg in memory.messages:
        st.chat_message(msg.type).markdown(msg.content)

    prompt = st.chat_input("Ask me anything about agriculture...")
    if prompt:
        with st.spinner("Thinking..."):
            try:
                # Classify the user's prompt first.
                classification = classify_query(prompt)

                if "no" in classification.lower():
                    final_translated_response = "I am Agri-Bot, your farming assistant. I can only answer questions related to agriculture."
                else:
                    translated_query, original_lang = translate_to_english(prompt)
                    chat_history = memory.messages
                    
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
                
                # --- THIS IS THE FIX ---
                # Manually save the conversation turn to memory in all cases.
                # This ensures the refusal message is also saved and displayed.
                memory.add_user_message(prompt)
                memory.add_ai_message(final_translated_response)
                # ---------------------
                
                # Rerun the app to display the updated chat history
                st.rerun()

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")

def main():
    """Main function to run the Streamlit app."""
    render_chat_ui()

if __name__ == "__main__":
    main()
