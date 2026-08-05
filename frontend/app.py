import streamlit as st
from api_client import query_backend, upload_document

# Page setup
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------------------------------------
# Sidebar: File Upload & Document Management
# ---------------------------------------------------------
st.sidebar.title("📄 Document Manager")
st.sidebar.caption("Upload files to automatically embed and index into Pinecone.")

uploaded_file = st.sidebar.file_uploader(
    "Choose a file to ingest",
    type=["pdf", "docx", "csv", "txt", "md"],
    help="Supported formats: PDF, Word (.docx), CSV, Text (.txt, .md)"
)

if uploaded_file is not None:
    if st.sidebar.button("Process & Index File", type="primary", use_container_width=True):
        with st.sidebar.spinner("Saving, chunking, and indexing into Pinecone..."):
            response = upload_document(uploaded_file)

            if "error" in response:
                st.sidebar.error(f"❌ {response['error']}")
            else:
                msg = response.get("message", "Success!")
                chunks = response.get("chunks_created", 0)
                st.sidebar.success(f"✅ {msg}")
                st.sidebar.info(f"Created **{chunks}** vector chunk(s).")

st.sidebar.divider()
st.sidebar.markdown(
    "**Supported formats:**\n"
    "- 📕 PDF (`.pdf`)\n"
    "- 📘 Word (`.docx`)\n"
    "- 🟢 CSV (`.csv`)\n"
    "- 📄 Text (`.txt`, `.md`)"
)

# ---------------------------------------------------------
# Main Chat Interface
# ---------------------------------------------------------
st.title("🤖 Document Assistant")
st.caption("Ask questions strictly derived from your uploaded documents.")

# Initialize session state for message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("📚 View Sources & Citations"):
                for src in message["sources"]:
                    st.markdown(f"- 📄 **{src['filename']}** ({src['page']})")

# User prompt input
if prompt := st.chat_input("Ask a question about the documents..."):
    # Render user query
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})

    # Render assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            response_data = query_backend(prompt)
            answer_text = response_data.get("answer", "")
            sources = response_data.get("sources", [])

            st.markdown(answer_text)

            if sources:
                with st.expander("📚 View Sources & Citations"):
                    for src in sources:
                        st.markdown(f"- 📄 **{src['filename']}** ({src['page']})")

    # Persist in session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "sources": sources
    })

