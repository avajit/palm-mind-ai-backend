import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.schemas import ChatRequest, ChatResponse
from app.services.rag import RAGService

router = APIRouter()

def get_rag() -> RAGService:
    """Dependency injection for RAGService."""
    return RAGService()

@router.post("/chat", response_model=ChatResponse, status_code=200)
async def chat_endpoint(
    request: ChatRequest,
    db: Session = Depends(get_db),
    rag: RAGService = Depends(get_rag),
) -> ChatResponse:
    """
    Conversational RAG API endpoint:
    1. Handles multi-turn chat questions using vector search & chat memory.
    2. Handles interview booking intent (name, email, date, time collection & validation).
    3. Stores completed bookings in SQLite and chat history in Redis.
    """
    session_id = request.session_id or str(uuid.uuid4())
    message = request.message.strip()

    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        result = rag.process_message(session_id=session_id, message=message, db=db)
        return ChatResponse(
            session_id=result["session_id"],
            intent=result["intent"],
            answer=result["answer"],
            sources=[s["filename"] for s in result["sources"]],
            booking=result["booking"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")
