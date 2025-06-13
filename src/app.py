# src/app.py

import streamlit as st
from rag_chat import generate_rag_response

st.set_page_config(page_title="📚 RAG Chatbot", layout="wide")

st.title("🧠 GAL-RAG Chatbot")
st.markdown("Ask any question based on your document knowledge base.")

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Chat input
user_input = st.chat_input("Ask a question...")

# Handle new question
if user_input:
    with st.spinner("Thinking..."):
        try:
            answer = generate_rag_response(user_input)
        except Exception as e:
            answer = f"❌ Error generating response: {str(e)}"
        st.session_state.history.append((user_input, answer))

# Display conversation
for q, a in st.session_state.history:
    st.chat_message("user").markdown(q)
    st.chat_message("ai").markdown(a)
