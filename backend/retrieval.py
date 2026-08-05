from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    TOP_K_RESULTS,
)


def get_retriever():
    """Initializes and returns a Pinecone vector store retriever using llama-text-embed-v2 embeddings."""
    embeddings = PineconeEmbeddings(
        model=EMBEDDING_MODEL,
        pinecone_api_key=PINECONE_API_KEY
    )

    vector_store = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS}
    )
    return retriever


if __name__ == "__main__":
    # Independent verification
    retriever = get_retriever()
    results = retriever.invoke("Test query to verify retrieval pipeline")
    print(f"[SUCCESS] Retrieved {len(results)} context chunk(s).")