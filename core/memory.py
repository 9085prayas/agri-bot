import streamlit as st
from langchain.memory import ConversationBufferMemory

def get_memory():
    if "chat_memory" not in st.session_state:
        st.session_state.chat_memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
    return st.session_state.chat_memory
