from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseVectorStore(ABC):
    """
    Abstract interface for Vector Store operations.
    Hides the underlying implementation (FAISS, Chroma, Pinecone, etc).
    """
    
    @abstractmethod
    def add(self, vectors: List[List[float]], metadatas: List[Dict[str, Any]]):
        """
        Add vectors and their associated metadata to the store.
        vectors and metadatas must have same length.
        """
        pass
    
    @abstractmethod
    def search(self, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        """
        Search for most similar vectors.
        Returns list of metadata dicts with added 'score' field.
        """
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Return total number of vectors in store"""
        pass
