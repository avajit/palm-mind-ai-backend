from typing import List, Protocol
import re
from app.db import ChunkingStrategy

class ChunkerProtocol(Protocol):
    def chunk(self, text: str) -> List[str]:
        ...

class FixedChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:
        """Splits text into fixed-size character windows with overlap."""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            # Move forward by chunk_size minus overlap, but ensure we always move forward
            step = max(1, self.chunk_size - self.overlap)
            start += step
            
        return chunks

class SentenceChunker:
    def __init__(self, max_chunk_size: int = 500, overlap_sentences: int = 1):
        self.max_chunk_size = max_chunk_size
        self.overlap_sentences = overlap_sentences

    def chunk(self, text: str) -> List[str]:
        """Splits text by sentences, grouping them up to max_chunk_size, with sentence overlap."""
        sentences = re.split(r'(?<=[.!?])[\s\n]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return [text[i:i+self.max_chunk_size] for i in range(0, len(text), self.max_chunk_size - 50)]

        chunks = []
        current_chunk: List[str] = []
        current_length = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            
            # If a single sentence is longer than max_chunk_size, split it into character sub-chunks
            if len(sentence) > self.max_chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                for sub_i in range(0, len(sentence), self.max_chunk_size - 50):
                    chunks.append(sentence[sub_i:sub_i + self.max_chunk_size])
                i += 1
                continue

            if current_length + len(sentence) > self.max_chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                overlap_slice = current_chunk[-self.overlap_sentences:] if self.overlap_sentences > 0 else []
                current_chunk = overlap_slice
                current_length = sum(len(s) + 1 for s in current_chunk)
            
            current_chunk.append(sentence)
            current_length += len(sentence) + 1
            i += 1
            
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

def get_chunker(strategy: ChunkingStrategy) -> ChunkerProtocol:
    """Factory function to get the requested chunker (Abstract Factory Pattern)."""
    if strategy == ChunkingStrategy.fixed:
        return FixedChunker()
    elif strategy == ChunkingStrategy.sentence:
        return SentenceChunker()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
