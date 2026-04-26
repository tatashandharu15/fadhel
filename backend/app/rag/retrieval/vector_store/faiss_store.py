import threading
import numpy as np
import logging
from typing import List, Dict, Any
from backend.app.rag.retrieval.vector_store.base import BaseVectorStore

logger = logging.getLogger(__name__)

class FaissVectorStore(BaseVectorStore):
    """
    In-Memory Vector Store implementation using FAISS.
    Uses IndexFlatIP (Inner Product) with normalized vectors for Cosine Similarity.
    Thread-safe implementation.
    """
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.metadatas: List[Dict[str, Any]] = []
        # Fallback to numpy storage if faiss fails or just use numpy directly
        self.vectors_list: List[np.ndarray] = [] 
        self._lock = threading.Lock()
        self._index = None
        
        logger.info(f"Simple Numpy VectorStore initialized with dim={dimension}")

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors to unit length for cosine similarity"""
        if vectors.size == 0:
            return vectors
        # L2 norm along axis 1
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1e-10
        return vectors / norms

    def add(self, vectors: List[List[float]], metadatas: List[Dict[str, Any]]):
        if len(vectors) != len(metadatas):
            raise ValueError("Vectors and metadatas must have same length")
        
        if not vectors:
            return

        # Convert to float32 numpy array
        vec_np = np.array(vectors, dtype=np.float32)
        
        # Check dimension
        if vec_np.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension mismatch. Expected {self.dimension}, got {vec_np.shape[1]}")
            
        # Normalize for cosine similarity
        vec_norm = self._normalize(vec_np)
        
        with self._lock:
            # Store vectors in list
            for v in vec_norm:
                self.vectors_list.append(v)
            # Add to metadata store
            self.metadatas.extend(metadatas)
            
        logger.debug(f"Added {len(vectors)} vectors to store. Total: {self.count()}")

    def search(self, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        print(f"[NumpyStore] Search called. Metadatas: {len(self.metadatas)}")
        if not self.metadatas:
            return []
            
        # Prepare query vector
        q_np = np.array([query_vector], dtype=np.float32)
        
        if q_np.shape[1] != self.dimension:
            raise ValueError(f"Query dimension mismatch. Expected {self.dimension}, got {q_np.shape[1]}")
            
        # Normalize query
        q_norm = self._normalize(q_np)
        # q_norm shape: (1, dim)
        
        print("[NumpyStore] Acquiring lock...")
        with self._lock:
            print("[NumpyStore] Lock acquired. Searching...")
            
            # Convert list of vectors to 2D array: (N, dim)
            if not self.vectors_list:
                 return []
            
            db_vectors = np.array(self.vectors_list)
            
            # Dot product: (N, dim) dot (dim, 1) -> (N, 1)
            # q_norm is (1, dim), so we transpose it or just dot
            # scores = db_vectors @ q_norm.T
            scores = np.dot(db_vectors, q_norm.flatten())
            
            # Get top k
            k = min(top_k, len(scores))
            # argsort returns indices of sorted elements (ascending)
            # we want descending
            top_indices = np.argsort(-scores)[:k]
            
            results = []
            for idx in top_indices:
                score = float(scores[idx])
                print(f"[NumpyStore] Candidate idx={idx} score={score}")
                
                meta = self.metadatas[idx].copy()
                meta['score'] = float(score)
                results.append(meta)
                
            print(f"[NumpyStore] Found {len(results)} results.")
            return results

    def count(self) -> int:
        return len(self.metadatas)

    def clear(self):
        with self._lock:
            self.vectors_list = []
            self.metadatas = []
