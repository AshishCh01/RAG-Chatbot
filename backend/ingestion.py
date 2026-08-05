import os
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from document_loader import load_all_documents


def ingest_documents():
    """Loads documents, splits into chunks, generates embeddings, and uploads to Pinecone."""
    print("--- STEP 1: Loading Documents ---")
    documents = load_all_documents()

    if not documents:
        print("[CANCELLED] No documents found to process. Ingestion stopped.")
        return

    print("\n--- STEP 2: Splitting Documents into Chunks ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"[SUCCESS] Split {len(documents)} document record(s) into {len(chunks)} chunk(s).")

    print("\n--- STEP 3: Initializing Pinecone & Index Check ---")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Fetch index list and create index if missing
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Index '{PINECONE_INDEX_NAME}' not found. Creating index...")
        try:
            # Create index configured for integrated model inference
            pc.create_index_for_model(
                name=PINECONE_INDEX_NAME,
                cloud="aws",
                region="us-east-1",
                embed={
                    "model": EMBEDDING_MODEL,
                    "field_map": {"text": "text"}
                }
            )
            print(f"[SUCCESS] Created integrated index '{PINECONE_INDEX_NAME}'.")
        except Exception:
            # Fallback to standard serverless index (llama-text-embed-v2 uses 1024 dimensions by default)
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"[SUCCESS] Created serverless index '{PINECONE_INDEX_NAME}' with 1024 dimensions.")
    else:
        print(f"[INFO] Index '{PINECONE_INDEX_NAME}' already exists.")

    print("\n--- STEP 4: Embedding & Upserting to Pinecone ---")
    embeddings = PineconeEmbeddings(
        model=EMBEDDING_MODEL,
        pinecone_api_key=PINECONE_API_KEY
    )

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME
    )

    print(f"\n[COMPLETE] Successfully ingested {len(chunks)} chunks into Pinecone index '{PINECONE_INDEX_NAME}'.")


if __name__ == "__main__":
    ingest_documents()