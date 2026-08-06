import os
from pathlib import Path
from pydantic import SecretStr
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from ocr_loader import load_image_with_ocr, OCR_SUPPORTED_EXTENSIONS
from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

DOCUMENTS_DIR = "./documents"


def load_single_document(file_path: str):
    """Loads a single document based on its file extension (.pdf, .docx, .csv, .txt, .md)."""
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    file_str = str(path)

    if ext == ".pdf":
        loader = PyPDFLoader(file_str)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_str)
    elif ext == ".csv":
        loader = CSVLoader(file_str, encoding="utf-8")
    elif ext in [".txt", ".md"]:
        loader = TextLoader(file_str, encoding="utf-8")
    elif ext in OCR_SUPPORTED_EXTENSIONS:
        return load_image_with_ocr(file_str)   # returns Documents directly, no .load() needed
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    return loader.load()


def ingest_single_file(file_path: str) -> int:
    """Loads, chunks, embeds, and uploads a single file to Pinecone."""
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is not configured.")

    print(f"\n[INGESTING] Processing single file: {file_path}")
    docs = load_single_document(file_path)

    # Chunk the document
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(docs)
    print(f"[INGESTING] Created {len(chunks)} chunk(s) for {file_path}")

    if not chunks:
        raise ValueError(
            f"No text could be extracted from '{Path(file_path).name}' "
            "(OCR may have found no readable text in the image)."
        )

    # Initialize Pinecone embeddings with SecretStr wrapping
    embeddings = PineconeEmbeddings(
        model=EMBEDDING_MODEL,
        pinecone_api_key=SecretStr(PINECONE_API_KEY)
    )

    # Upload chunks to Pinecone index
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME
    )
    print(f"[SUCCESS] Uploaded {len(chunks)} chunk(s) from {file_path} to Pinecone!")
    return len(chunks)


def run_ingestion():
    """Batch loads and ingests all supported files inside the documents directory."""
    dir_path = Path(DOCUMENTS_DIR)
    if not dir_path.exists():
        print(f"[ERROR] Directory '{DOCUMENTS_DIR}' does not exist.")
        return

    supported_extensions = {".pdf", ".docx", ".csv", ".txt", ".md"} | OCR_SUPPORTED_EXTENSIONS
    files = [f for f in dir_path.iterdir() if f.is_file() and f.suffix.lower() in supported_extensions]

    if not files:
        print("[INFO] No supported document files found.")
        return

    print(f"Found {len(files)} document file(s) for batch ingestion.")
    total_chunks = 0
    for f in files:
        try:
            total_chunks += ingest_single_file(str(f))
        except Exception as e:
            print(f"[ERROR] Failed to ingest {f.name}: {e}")

    print(f"\n[COMPLETE] Batch ingestion finished. Total chunks indexed: {total_chunks}")


if __name__ == "__main__":
    run_ingestion()


