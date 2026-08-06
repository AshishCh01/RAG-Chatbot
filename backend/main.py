import shutil
from pathlib import Path
from typing import List, Dict, Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic import BaseModel
from generation import answer_question, answer_image_question
from ingestion import ingest_single_file, DOCUMENTS_DIR
from ocr_loader import OCR_SUPPORTED_EXTENSIONS

app = FastAPI(title="RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".csv", ".txt", ".md"} | OCR_SUPPORTED_EXTENSIONS


class MessageItem(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    question: str
    chat_history: Optional[List[MessageItem]] = []


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/query")
def query_documents(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        history_list = [item.model_dump() for item in request.chat_history] if request.chat_history else []
        result = answer_question(request.question, history_list)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
def upload_document(file: UploadFile = File(...)):
    """Receives an uploaded document, saves it to disk, and triggers vector ingestion."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    file_path = Path(file.filename)
    ext = file_path.suffix.lower()

    # 1. Validate file format
    if ext not in ALLOWED_EXTENSIONS:
        allowed_str = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported formats: {allowed_str}"
        )

    # 2. Ensure destination directory exists
    dest_dir = Path(DOCUMENTS_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)

    save_path = dest_dir / file.filename

    # 3. Save incoming file to local documents directory
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # 4. Trigger vector embedding and ingestion for the uploaded file
    try:
        chunks_count = ingest_single_file(str(save_path))
        return {
            "message": "File uploaded and ingested successfully!",
            "filename": file.filename,
            "chunks_created": chunks_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@app.post("/query-image")
def query_with_image(
    file: UploadFile = File(...),
    question: Optional[str] = Form(None),
):
    """Receives an image attached in the chat input, OCRs it on the spot, and answers
    the question WITHOUT saving anything to Pinecone. If `question` is empty, the OCR'd
    text from the image itself is treated as the question."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No image provided.")

    ext = Path(file.filename).suffix.lower()
    if ext not in OCR_SUPPORTED_EXTENSIONS:
        allowed_str = ", ".join(sorted(OCR_SUPPORTED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format '{ext}'. Supported formats: {allowed_str}"
        )

    try:
        image_bytes = file.file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read image: {str(e)}")

    try:
        result = answer_image_question(image_bytes, file.filename, question)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))