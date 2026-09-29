from sentence_transformers import SentenceTransformer
from typing import List
import torch

class EmbeddingService:
    """
    Singleton service to handle embeddings.
    We load the machine learning model only once when it's first needed.
    """
    _model = None

    @classmethod
    def get_model(cls) -> SentenceTransformer:
        if cls._model is None:
            # Limit PyTorch CPU threads to prevent maxing out system CPU
            torch.set_num_threads(2)
            # all-MiniLM-L6-v2 is a highly efficient model that outputs 384 dimensions
            cls._model = SentenceTransformer("all-MiniLM-L6-v2")
        return cls._model

    @classmethod
    def generate_embeddings(cls, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of text strings."""
        model = cls.get_model()
        # encode returns numpy arrays; we convert to standard python lists for Qdrant
        embeddings = model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
