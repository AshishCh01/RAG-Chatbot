import streamlit as st
from api_client import query_backend

# Page configuration
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Document Assistant")
st.caption("Ask questions strictly derived from your uploaded documents.")

# Initialize chat message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User query input box
if prompt := st.chat_input("Ask a question about the documents..."):
    # Render user prompt
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Query API and render response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            response_text = query_backend(prompt)
            st.markdown(response_text)

    st.session_state.messages.append({"role": "assistant", "content": response_text})