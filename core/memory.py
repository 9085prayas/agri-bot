import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

def get_memory():
    """
    Returns a StreamlitChatMessageHistory object.
    This is the modern, recommended way to handle memory in Streamlit apps.
    The history is automatically persisted in st.session_state.
    """
    return StreamlitChatMessageHistory(key="langchain_messages")