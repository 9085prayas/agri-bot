import streamlit as st
from agent.conversational import get_conversational_agent
from core.translation import translate_to_english, translate_back
from core.memory import get_memory

def render_chat_ui():
    st.title("🌱 Agri-Bot")
    st.subheader("Your Farming Assistant")

    if st.button("New Conversation"):
        get_memory().clear()
        st.session_state.messages = []
        st.success("Chat history cleared!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display past messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    prompt = st.chat_input("Type here (any language)...")
    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            translated_query, original_lang = translate_to_english(prompt)
            st.write(f"🔍 Language: {original_lang.upper()}")
            st.write(f"🔄 Translated: {translated_query}")

            agent = get_conversational_agent()

            # Instruction prompt
            full_prompt = f"""
You are AgriBot — a trusted digital farming assistant.

RULES:
1) Only answer farming-related questions (crops, soil, irrigation, fertilizers, weather, govt schemes).
2) If irrelevant, reply: "⚠ Please ask questions related to agriculture only."
3) Use external tools (Wikipedia, Arxiv, Tavily) at most twice.
4) Keep answers short, clear, and farmer-friendly.

User’s Question (English): {translated_query}
"""

            response = agent.invoke({"input": full_prompt})
            response_text = response.get("output", str(response))
            final_response = translate_back(response_text, original_lang)

            st.chat_message("assistant").markdown(final_response)
            st.session_state.messages.append({"role": "assistant", "content": final_response})

        except Exception as e:
            st.error(f"Error: {e}")
