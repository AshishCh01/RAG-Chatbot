
import streamlit as st
from api_client import query_backend

# Page setup
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Document Assistant")
st.caption("Ask questions strictly derived from your uploaded documents.")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history with sources expandable block
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display source citations if available
        if message.get("sources"):
            with st.expander("📚 View Sources & Citations"):
                for src in message["sources"]:
                    st.markdown(f"- 📄 **{src['filename']}** ({src['page']})")

# User query handling
if prompt := st.chat_input("Ask a question about the documents..."):
    # Display user input in UI
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})

    # Call FastAPI backend and render response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            response_data = query_backend(prompt)
            answer_text = response_data.get("answer", "")
            sources = response_data.get("sources", [])

            # Render generated answer text
            st.markdown(answer_text)

            # Render citation dropdown
            if sources:
                with st.expander("📚 View Sources & Citations"):
                    for src in sources:
                        st.markdown(f"- 📄 **{src['filename']}** ({src['page']})")

    # Save answer and sources to Streamlit chat session
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "sources": sources
    })