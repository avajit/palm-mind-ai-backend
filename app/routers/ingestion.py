from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.orm import Session

from app.db import get_db, DocumentRecord, ChunkingStrategy
from app.schemas import DocumentResponse
from app.services.extraction import extract_text
from app.services.chunking import get_chunker
from app.services.embeddings import EmbeddingService
from app.services.vector_store import QdrantService

router = APIRouter()

# Max file size: 10 MB
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024


def get_qdrant() -> QdrantService:
    """Dependency injection for QdrantService."""
    return QdrantService()


@router.post("/ingest", response_model=DocumentResponse, status_code=200)
async def ingest_document(
    file: UploadFile = File(...),
    chunking_strategy: ChunkingStrategy = Form(ChunkingStrategy.fixed),
    db: Session = Depends(get_db),
    qdrant: QdrantService = Depends(get_qdrant),
) -> DocumentResponse:
    """
    Ingests a PDF or TXT file:
    1. Validates file type and size.
    2. Extracts raw text.
    3. Chunks the text using the selected strategy.
    4. Generates embeddings for each chunk.
    5. Upserts chunks + vectors into Qdrant.
    6. Saves document metadata into SQLite.
    7. Returns document_id, chunk_count, and strategy used.
    """

    # --- Step 1: Validate file type ---
    filename = file.filename or ""
    if not filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Only .pdf and .txt files are allowed.",
        )

    # --- Step 2: Read file bytes & validate size ---
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE_BYTES // (1024*1024)} MB.",
        )

    # --- Step 3: Extract text ---
    try:
        text = extract_text(file_bytes, filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")

    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail="The file appears to be empty or contains no extractable text.",
        )

    # --- Step 4: Chunk the text ---
    chunker = get_chunker(chunking_strategy)
    chunks: list[str] = chunker.chunk(text)

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="Chunking produced no output. The document may be too short.",
        )

    # --- Step 5: Save metadata to SQLite FIRST to get the document_id ---
    doc_record = DocumentRecord(
        filename=filename,
        file_type=".pdf" if filename.lower().endswith(".pdf") else ".txt",
        strategy=chunking_strategy,
        chunk_count=len(chunks),
        char_count=len(text),
    )
    db.add(doc_record)
    db.commit()
    db.refresh(doc_record)  # Populates doc_record.id

    # --- Step 6: Generate embeddings ---
    try:
        embeddings: list[list[float]] = EmbeddingService.generate_embeddings(chunks)
    except Exception as e:
        # Rollback the DB record if embedding fails
        db.delete(doc_record)
        db.commit()
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

    # --- Step 7: Upsert into Qdrant ---
    try:
        qdrant.upsert_chunks(
            document_id=doc_record.id,
            filename=filename,
            chunks=chunks,
            embeddings=embeddings,
        )
    except Exception as e:
        # Rollback the DB record if Qdrant upsert fails
        db.delete(doc_record)
        db.commit()
        raise HTTPException(status_code=500, detail=f"Vector store upsert failed: {str(e)}")

    # --- Step 8: Return response ---
    return DocumentResponse(
        document_id=doc_record.id,
        chunk_count=len(chunks),
        strategy=chunking_strategy.value,
    )
