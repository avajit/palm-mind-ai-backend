from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from app.config import settings
import uuid
from typing import List

class QdrantService:
    """Service to handle all interactions with the Qdrant Vector Database."""
    
    def __init__(self):
        # Connect to our local Docker Qdrant instance
        self.client = QdrantClient(url=settings.qdrant_url)
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
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )
        return results
