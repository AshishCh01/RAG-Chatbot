from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from generation import answer_question

app = FastAPI(
    title="RAG Chatbot API",
    description="Backend API serving Google Gemini 2.5 Flash & Pinecone RAG operations",
    version="1.0.0"
)

# Enable CORS for local frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


@app.get("/health")
def health_check():
    """Health check endpoint to verify backend status."""
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    """Processes user question through the RAG pipeline."""
    user_query = request.question.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Question string cannot be empty.")

    try:
        answer = answer_question(user_query)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while generating the response: {str(e)}"
        )