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
    status: Optional[str] = None
    name: Optional[str] = None
    email: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    missing: Optional[List[str]] = None
    errors: Optional[List[str]] = None
    booking_id: Optional[int] = None

class ChatResponse(BaseModel):
    session_id: str
    intent: str
    answer: str
    sources: List[str] = []
    booking: Optional[BookingDetails] = None
