# from typing import List
# from langchain_core.documents import Document
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain_google_genai import ChatGoogleGenerativeAI

# from config import GOOGLE_API_KEY, LLM_MODEL
# from retrieval import get_retriever


# def format_docs(docs: List[Document]) -> str:
#     """Formats retrieved document chunks into a single concatenated string."""
#     if not docs:
#         return "No relevant context found."
#     return "\n\n---\n\n".join([doc.page_content for doc in docs])


# def get_rag_chain():
#     """Builds the LCEL RAG chain combining vector retrieval, prompt formatting, and Gemini 2.0 Flash."""
#     retriever = get_retriever()

#     # Initialize Google Gemini 2.5 Flash model
#     llm = ChatGoogleGenerativeAI(
#         model=LLM_MODEL,
#         google_api_key=GOOGLE_API_KEY,
#         temperature=0.0,
#         max_retries=5,
#     )

#     # System prompt enforcing strict context constraints and required fallback string
#     system_prompt = (
#         "You are a strict QA assistant. Answer questions based ONLY on the provided context.\n"
#         "Rules:\n"
#         "1. Do NOT use any prior external knowledge or assumptions.\n"
#         "2. If the answer cannot be directly derived from the context below, reply EXACTLY with:\n"
#         "   \"I couldn't find that information in the uploaded documents.\"\n"
#         "3. Keep the answer concise, factual, and strictly faithful to the text.\n\n"
#         "Context:\n"
#         "{context}\n\n"
#         "Question: {question}"
#     )

#     prompt = ChatPromptTemplate.from_template(system_prompt)

#     # LangChain Expression Language (LCEL) pipeline
#     rag_chain = (
#         {"context": retriever | format_docs, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

#     return rag_chain


# def answer_question(query: str) -> str:
#     """Executes the RAG chain for an incoming prompt and returns the answer string."""
#     chain = get_rag_chain()
#     return chain.invoke(query)


# if __name__ == "__main__":
#     # Test generation pipeline independently
#     test_query = "What is discussed in the documents?"
#     response = answer_question(test_query)
#     print("Response:\n", response)



import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import LLM_MODEL, GOOGLE_API_KEY
from retrieval import get_retriever

SYSTEM_PROMPT = """You are a helpful assistant for document questioning.
Answer the user's question strictly derived ONLY from the provided context below.
If the answer cannot be found in the context, respond EXACTLY with:
"I couldn't find that information in the uploaded documents."

Context:
{context}
"""


def answer_question(query: str) -> dict:
    """Retrieves relevant context, extracts metadata sources, generates answer, and returns structured payload."""
    retriever = get_retriever()

    # 1. Retrieve relevant text chunks with metadata
    docs = retriever.invoke(query)

    # 2. Extract and format sources (deduplicated)
    raw_sources = []
    context_texts = []

    for doc in docs:
        context_texts.append(doc.page_content)

        # Extract filename from absolute file path
        source_path = doc.metadata.get("source", "Unknown Document")
        filename = os.path.basename(source_path)

        # Convert 0-indexed page numbers to 1-indexed for display
        page = doc.metadata.get("page")
        if page is not None:
            try:
                page_str = f"Page {int(page) + 1}"
            except (ValueError, TypeError):
                page_str = f"Page {page}"
        else:
            page_str = "Page N/A"

        raw_sources.append({"filename": filename, "page": page_str})

    # Deduplicate sources while keeping original order
    unique_sources = []
    seen = set()
    for src in raw_sources:
        pair = (src["filename"], src["page"])
        if pair not in seen:
            seen.add(pair)
            unique_sources.append(src)

    # 3. Format prompt and invoke Gemini model with StrOutputParser
    context = "\n\n---\n\n".join(context_texts)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.0,
        max_retries=5
    )

    # StrOutputParser forces the output to be a plain string
    chain = prompt | llm | StrOutputParser()
    answer_text = chain.invoke({"context": context, "question": query})

    # Clear citation list if the strict fallback message was triggered
    if "couldn't find that information" in answer_text.lower():
        sources_to_return = []
    else:
        sources_to_return = unique_sources

    return {
        "answer": answer_text,
        "sources": sources_to_return
    }


if __name__ == "__main__":
    # Test execution locally
    test_result = answer_question("How many layers are in the encoder stack?")
    print("Answer:", test_result["answer"])
    print("Sources:", test_result["sources"])