from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseVectorStore(ABC):
    """
    Interface untuk operasi Vector Database.
    Memudahkan penggantian DB (e.g., dari FAISS ke Qdrant/Pinecone).
    """

    @abstractmethod
    async def add_documents(self, documents: List[Dict[str, Any]]):
        """Menambah dokumen ke vector store"""
        pass

    @abstractmethod
    async def search(self, query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Mencari dokumen relevan"""
        pass

class BaseRetriever(ABC):
    """
    Component high-level yang menggunakan VectorStore.
    """
    @abstractmethod
    async def retrieve(self, query: str) -> str:
        """Mengembalikan context string"""
        pass
