from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingService:
    """
    Singleton service to handle embeddings.
    We load the machine learning model only once when it's first needed.
    """
    _model = None

    @classmethod
    def get_model(cls) -> SentenceTransformer:
        if cls._model is None:
            # all-MiniLM-L6-v2 is a highly efficient model that outputs 384 dimensions
            cls._model = SentenceTransformer("all-MiniLM-L6-v2")
        return cls._model

    @classmethod
    def generate_embeddings(cls, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of text strings."""
        model = cls.get_model()
        # encode returns numpy arrays; we convert to standard python lists for Qdrant
        embeddings = model.encode(texts)
        return embeddings.tolist()
