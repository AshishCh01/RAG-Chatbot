from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from config import GOOGLE_API_KEY, LLM_MODEL
from retrieval import get_retriever


def format_docs(docs: List[Document]) -> str:
    """Formats retrieved document chunks into a single concatenated string."""
    if not docs:
        return "No relevant context found."
    return "\n\n---\n\n".join([doc.page_content for doc in docs])


def get_rag_chain():
    """Builds the LCEL RAG chain combining vector retrieval, prompt formatting, and Gemini 2.0 Flash."""
    retriever = get_retriever()

    # Initialize Google Gemini 2.5 Flash model
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.0,
        max_retries=5,
    )

    # System prompt enforcing strict context constraints and required fallback string
    system_prompt = (
        "You are a strict QA assistant. Answer questions based ONLY on the provided context.\n"
        "Rules:\n"
        "1. Do NOT use any prior external knowledge or assumptions.\n"
        "2. If the answer cannot be directly derived from the context below, reply EXACTLY with:\n"
        "   \"I couldn't find that information in the uploaded documents.\"\n"
        "3. Keep the answer concise, factual, and strictly faithful to the text.\n\n"
        "Context:\n"
        "{context}\n\n"
        "Question: {question}"
    )

    prompt = ChatPromptTemplate.from_template(system_prompt)

    # LangChain Expression Language (LCEL) pipeline
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def answer_question(query: str) -> str:
    """Executes the RAG chain for an incoming prompt and returns the answer string."""
    chain = get_rag_chain()
    return chain.invoke(query)


if __name__ == "__main__":
    # Test generation pipeline independently
    test_query = "What is discussed in the documents?"
    response = answer_question(test_query)
    print("Response:\n", response)