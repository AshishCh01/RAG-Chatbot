import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file in the backend directory
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Pinecone Configuration
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-index")
# Model identifier for Pinecone integrated inference embedding
EMBEDDING_MODEL = "llama-text-embed-v2"

# Gemini LLM Configuration
LLM_MODEL = "gemini-flash-latest"

# Document & Ingestion Settings
DOCUMENTS_DIR = Path(__file__).resolve().parent / "documents"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150 
TOP_K_RESULTS = 4

# Validate required variables
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing in backend/.env")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing in backend/.env")