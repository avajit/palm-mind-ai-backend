from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from app.config import settings
import uuid
from typing import List

class QdrantService:
    """Service to handle all interactions with the Qdrant Vector Database."""
    
    def __init__(self):
        # Connect to our local Docker Qdrant instance with a robust timeout
        self.client = QdrantClient(url=settings.qdrant_url, timeout=30)
        self.collection_name = "documents"
        self.ensure_collection()

    def ensure_collection(self):
        """Creates the collection if it doesn't already exist."""
        # 384 is the exact dimension of our all-MiniLM-L6-v2 embeddings
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

    def upsert_chunks(self, document_id: int, filename: str, chunks: List[str], embeddings: List[List[float]]):
        """Saves the text chunks and their vectors into Qdrant."""
        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()), # Unique ID for each chunk
                    vector=vector,
                    payload={
                        "document_id": document_id,
                        "filename": filename,
                        "chunk_index": i,
                        "text": chunk
                    }
                )
            )
        # Uploads all points in one batch to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(self, query_vector: List[float], limit: int = 3):
        """Searches for the most similar chunks based on a query vector."""
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit
        )
        return [
            {
                "id": p.id,
                "score": p.score,
                "document_id": p.payload.get("document_id") if p.payload else None,
                "filename": p.payload.get("filename") if p.payload else None,
                "chunk_index": p.payload.get("chunk_index") if p.payload else None,
                "text": p.payload.get("text") if p.payload else None,
            }
            for p in response.points
        ]
