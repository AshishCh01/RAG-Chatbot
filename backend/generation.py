import os
from typing import Optional, List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import LLM_MODEL, GOOGLE_API_KEY
from retrieval import get_retriever
from ocr_loader import extract_text_from_image_bytes

SYSTEM_PROMPT = """You are a helpful assistant for document questioning.
Answer the user's question strictly derived ONLY from the provided context below.
If the answer cannot be found in the context, respond EXACTLY with:
"I couldn't find that information in the uploaded documents."

Context:
{context}
"""


def contextualize_question(query: str, chat_history: Optional[List[Dict[str, Any]]] = None) -> str:
    """Rewrites follow-up questions into standalone queries using recent chat history."""
    if not chat_history:
        return query

    formatted_history = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in chat_history[-6:]]
    )

    rephrase_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the following chat history and a follow-up question, rephrase the follow-up question into a standalone question that can be fully understood without reading the history. Do NOT answer the question, only rewrite it. If it is already a standalone question, return it as is."),
        ("human", "Chat History:\n{chat_history}\n\nFollow-up Question: {question}\nStandalone Question:")
    ])

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.0,
        max_retries=3
    )

    chain = rephrase_prompt | llm | StrOutputParser()

    try:
        standalone_query = chain.invoke({"chat_history": formatted_history, "question": query}).strip()
        return standalone_query if standalone_query else query
    except Exception:
        return query


def _generate_answer(question: str, context: str) -> str:
    """Formats the prompt and invokes Gemini to produce a final answer from context."""
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

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


def _extract_sources(docs) -> List[Dict[str, str]]:
    """Extracts and deduplicates filename/page metadata from retrieved chunks."""
    raw_sources = []
    for doc in docs:
        source_path = doc.metadata.get("source", "Unknown Document")
        filename = os.path.basename(source_path)

        page = doc.metadata.get("page")
        if page is not None:
            try:
                page_str = f"Page {int(page) + 1}"
            except (ValueError, TypeError):
                page_str = f"Page {page}"
        else:
            page_str = "Page N/A"

        raw_sources.append({"filename": filename, "page": page_str})

    unique_sources = []
    seen = set()
    for src in raw_sources:
        pair = (src["filename"], src["page"])
        if pair not in seen:
            seen.add(pair)
            unique_sources.append(src)
    return unique_sources


def answer_question(query: str, chat_history: Optional[List[Dict[str, Any]]] = None) -> dict:
    """Retrieves relevant context, extracts metadata sources, generates answer, and returns structured payload."""
    retriever = get_retriever()
    history = chat_history or []

    # 1. Reformulate follow-up query into a standalone query for Pinecone
    search_query = contextualize_question(query, history)

    # --------------------------------------------------
    # DEBUG LOGS (Check your FastAPI terminal)
    # --------------------------------------------------
    print("\n" + "=" * 40)
    print(f"📥 Original User Query : {query}")
    print(f"🔄 Rewritten Query Used : {search_query}")
    print("=" * 40 + "\n")
    # --------------------------------------------------

    # 2. Retrieve relevant text chunks using the standalone query
    docs = retriever.invoke(search_query)

    # 3. Extract and format sources (deduplicated)
    context_texts = [doc.page_content for doc in docs]
    unique_sources = _extract_sources(docs)

    # 4. Format prompt and invoke Gemini model
    context = "\n\n---\n\n".join(context_texts)
    answer_text = _generate_answer(query, context)

    if "couldn't find that information" in answer_text.lower():
        sources_to_return = []
    else:
        sources_to_return = unique_sources

    return {
        "answer": answer_text,
        "sources": sources_to_return
    }


def answer_image_question(
    image_bytes: bytes,
    filename: str,
    typed_question: Optional[str] = None,
    chat_history: Optional[List[Dict[str, Any]]] = None
) -> dict:
    """Handles an image attached directly in chat (NOT saved to Pinecone).
    OCRs the image, then either:
      - treats the OCR'd text AS the question (if user typed nothing), or
      - treats the OCR'd text as extra context alongside the user's typed question.
    Still retrieves supporting context from Pinecone, same as a normal text query.
    """
    ocr_text = extract_text_from_image_bytes(image_bytes, filename)
    if not ocr_text.strip():
        raise ValueError(f"No readable text found in '{filename}'.")

    history = chat_history or []

    if typed_question and typed_question.strip():
        question = typed_question.strip()
        image_context_note = f"[Text extracted from attached image '{filename}']:\n{ocr_text}"
    else:
        question = ocr_text
        image_context_note = None

    # Reformulate using chat history, same as the normal text flow
    search_query = contextualize_question(question, history)

    print("\n" + "=" * 40)
    print(f"🖼️ Image Attached : {filename}")
    print(f"📥 Question Used : {question}")
    print(f"🔄 Rewritten Query Used : {search_query}")
    print("=" * 40 + "\n")

    retriever = get_retriever()
    docs = retriever.invoke(search_query)

    context_texts = [doc.page_content for doc in docs]
    if image_context_note:
        context_texts.insert(0, image_context_note)

    unique_sources = _extract_sources(docs)

    context = "\n\n---\n\n".join(context_texts)
    answer_text = _generate_answer(question, context)

    if "couldn't find that information" in answer_text.lower():
        sources_to_return = []
    else:
        sources_to_return = unique_sources

    return {
        "answer": answer_text,
        "sources": sources_to_return,
        "extracted_text": ocr_text,
        "question_used": question
    }


if __name__ == "__main__":
    test_result = answer_question("How many layers are in the encoder stack?")
    print("Answer:", test_result["answer"])
    print("Sources:", test_result["sources"])

