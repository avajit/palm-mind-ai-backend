from pydantic import BaseModel, EmailStr
from typing import Optional, List
from .db import ChunkingStrategy

# --- Ingestion Schemas ---
class DocumentResponse(BaseModel):
    document_id: int
    chunk_count: int
    strategy: str

# --- Chat/Booking Schemas ---
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str

class BookingDetails(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    date: Optional[str] = None
    time: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    intent: str
    answer: str
    sources: List[str] = []
    booking: Optional[BookingDetails] = None
