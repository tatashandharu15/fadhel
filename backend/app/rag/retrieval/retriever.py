from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from backend.app.rag.retrieval.config import RetrievalConfig
from backend.app.rag.retrieval.vector_store.base import BaseVectorStore
from backend.app.rag.retrieval.vector_store.faiss_store import FaissVectorStore
# Import Embedder for seeding real vectors
from backend.app.rag.retrieval.embedder import HuggingFaceEmbedder

class BaseRetriever(ABC):
    @abstractmethod
    async def retrieve(self, query_vector: List[float], config: RetrievalConfig, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def add_documents(self, vectors: List[List[float]], documents: List[Dict[str, Any]]):
        """Allow adding documents to internal store"""
        pass

class InMemoryRetriever(BaseRetriever):
    """
    Retriever yang menggunakan In-Memory Vector Store (FAISS).
    Implements Singleton pattern to share state across API requests.
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InMemoryRetriever, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # Initialize Vector Store
        self.vector_store: BaseVectorStore = FaissVectorStore()
        
        # Pre-seed with dummy data if empty (for testing flow)
        if self.vector_store.count() == 0:
            self._seed_dummy_data()
            
        self._initialized = True
            
    def _seed_dummy_data(self):
        """Seed dummy vectors for testing (using real embedder)"""
        print("[Retriever] Seeding dummy data with REAL embeddings...")
        
        # 1. Define Dummy Documents
        docs = [
            {
                "id": "doc_101",
                "metadata": {"type": "spec_sheet", "vehicle": "Honda CR-V", "year": 2024},
                "payload": {"title": "Honda CR-V 2024 Engine Specs", "content": "Honda CR-V 2024 dibekali mesin 1.5L VTEC Turbo. Tenaga maksimum: 190 PS pada 6.000 rpm. Torsi maksimum: 240 Nm pada 1.700-5.000 rpm. Transmisi: CVT dengan Earth Dreams Technology."}
            },
            {
                "id": "doc_102",
                "metadata": {"type": "spec_sheet", "vehicle": "Honda CR-V", "year": 2024},
                "payload": {"title": "Honda CR-V 2024 Hybrid Specs", "content": "Varian CR-V e:HEV (Hybrid) menggunakan mesin 2.0L i-VTEC dipadukan dengan 2 motor listrik. Total tenaga sistem mencapai 207 PS dan torsi 335 Nm."}
            },
            {
                "id": "doc_201",
                "metadata": {"type": "spec_sheet", "vehicle": "Toyota Fortuner", "year": 2024},
                "payload": {"title": "Toyota Fortuner Specs", "content": "Toyota Fortuner 2.8 GR Sport menggunakan mesin diesel 1GD-FTV 2.755 cc yang menghasilkan tenaga 203.9 PS dan torsi 50.9 Kgm (sekitar 500 Nm)."}
            }
        ]
        
        # 2. Generate Vectors using Real Embedder
        try:
            embedder = HuggingFaceEmbedder()
            texts = [d["payload"]["content"] for d in docs]
            vectors = embedder.embed_batch(texts)
            
            # 3. Add to Vector Store
            self.vector_store.add(vectors, docs)
            print(f"[Retriever] Successfully seeded {len(docs)} documents to FAISS.")
            
        except Exception as e:
            print(f"[Retriever] Failed to seed dummy data: {e}")
            # Fallback to empty if embedding fails (should not happen in normal flow)

    async def retrieve(self, query_vector: List[float], config: RetrievalConfig, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Real vector search using FAISS
        """
        print(f"[Retriever] Searching with threshold {config.similarity_threshold}...")
        # Search in vector store
        results = self.vector_store.search(query_vector, config.top_k)
        
        # Filter results based on threshold
        filtered = []
        for r in results:
            print(f"[Retriever] Candidate: {r['id']} Score: {r['score']}")
            if r['score'] >= config.similarity_threshold:
                filtered.append(r)
        
        print(f"[Retriever] Found {len(filtered)} results after filtering.")
        return filtered

    def add_documents(self, vectors: List[List[float]], documents: List[Dict[str, Any]]):
        print(f"[Retriever] Adding {len(documents)} documents to store.")
        self.vector_store.add(vectors, documents)
        print(f"[Retriever] Total documents in store: {self.vector_store.count()}")
