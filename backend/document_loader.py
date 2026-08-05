import os
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
)

from config import DOCUMENTS_DIR


def load_single_document(file_path: Path) -> List[Document]:
    """Routes a single file to its appropriate LangChain loader based on extension."""
    ext = file_path.suffix.lower()
    str_path = str(file_path)

    try:
        if ext == ".pdf":
            loader = PyPDFLoader(str_path)
        elif ext == ".txt":
            loader = TextLoader(str_path, encoding="utf-8")
        elif ext == ".csv":
            loader = CSVLoader(str_path, encoding="utf-8")
        elif ext == ".docx":
            loader = Docx2txtLoader(str_path)
        else:
            print(f"[SKIP] Unsupported file format: {file_path.name}")
            return []

        docs = loader.load()
        print(f"[SUCCESS] Loaded {len(docs)} document record(s) from {file_path.name}")
        return docs

    except Exception as e:
        print(f"[ERROR] Failed to process {file_path.name}: {e}")
        return []


def load_all_documents(docs_directory: Path = DOCUMENTS_DIR) -> List[Document]:
    """Iterates through backend/documents/ and loads all supported files."""
    if not docs_directory.exists():
        raise FileNotFoundError(f"Documents directory does not exist: {docs_directory}")

    all_documents: List[Document] = []
    files = [f for f in docs_directory.iterdir() if f.is_file()]

    if not files:
        print(f"[WARN] No files found in {docs_directory}")
        return all_documents

    print(f"Found {len(files)} file(s) in {docs_directory}. Starting document loading...")

    for file_path in files:
        docs = load_single_document(file_path)
        all_documents.extend(docs)

    print(f"\nTotal document pages/records loaded: {len(all_documents)}")
    return all_documents


if __name__ == "__main__":
    # Independent execution check
    documents = load_all_documents()