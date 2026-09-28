from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.services.extraction import extract_text
from app.services.chunking import get_chunker
from app.db import ChunkingStrategy
from app.services.embeddings import EmbeddingService
from fastapi import Body

router = APIRouter()

@router.post("/test-extract")
async def test_extract_text(
    file: UploadFile = File(...),
    strategy: ChunkingStrategy = Form(ChunkingStrategy.fixed)
):
    """
    Temporary endpoint to test text extraction AND chunking.
    Upload a PDF or TXT file and choose a chunking strategy.
    """
    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")
    
    content = await file.read()
    try:
        # 1. Extract
        text = extract_text(content, file.filename)
        
        # 2. Chunk
        chunker = get_chunker(strategy)
        chunks = chunker.chunk(text)
        
        return {
            "filename": file.filename, 
            "strategy_used": strategy,
            "total_characters": len(text),
            "total_chunks": len(chunks),
            "first_chunk_preview": chunks[0][:200] + "..." if chunks else ""
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")



@router.post("/test-embed")
async def test_embedding(text: str = Body(..., embed=True)):
    """
    Temporary endpoint to test the Embedding Service.
    It will convert the input text into a vector and return its dimensions.
    """
    try:
        embeddings = EmbeddingService.generate_embeddings([text])
        vector = embeddings[0]
        return {
            "input_text": text,
            "vector_dimensions": len(vector),
            "first_five_numbers": vector[:5]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
