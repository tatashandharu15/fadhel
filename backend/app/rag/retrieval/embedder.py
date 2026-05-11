from abc import ABC, abstractmethod
from typing import List, Optional
import threading
import logging
import hashlib
import re
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

class BaseEmbedder(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed single string to vector"""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed list of strings to vectors"""
        pass

class HuggingFaceEmbedder(BaseEmbedder):
    """
    Real implementation using Sentence-Transformers.
    Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    Lightweight, CPU-friendly, deterministic, Multilingual.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self._model = None
        self._fallback_mode = False
        self._lock = threading.Lock()

    def _fallback_embed(self, text: str) -> List[float]:
        vector = np.zeros(384, dtype=np.float32)
        tokens = re.findall(r"[\w-]+", text.lower())
        if not tokens:
            return vector.tolist()

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % 384
            vector[index] += 1.0

        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector.tolist()
        
    def _ensure_model(self):
        """Lazy load model in a thread-safe way"""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        logger.info(f"Loading embedding model: {self.model_name}...")
                        print(f"DEBUG: Starting SentenceTransformer init for {self.model_name}...", flush=True)
                        from sentence_transformers import SentenceTransformer
                        # force cpu if needed, but library handles it well usually.
                        # Forcing CPU for stability in this environment
                        self._model = SentenceTransformer(self.model_name, device='cpu')
                        print(f"DEBUG: SentenceTransformer init complete.", flush=True)
                        logger.info("Embedding model loaded successfully.")
                    except ImportError:
                        logger.warning("sentence-transformers library not found. Falling back to deterministic hash embeddings.")
                        self._fallback_mode = True
                    except Exception as e:
                        logger.warning(f"Failed to load model {self.model_name}: {str(e)}. Falling back to deterministic hash embeddings.")
                        self._fallback_mode = True
        return self._model

    def embed_text(self, text: str) -> List[float]:
        if not text:
            # Handle empty input safely, return zero vector of correct dim (384)
            # all-MiniLM-L6-v2 dimension is 384
            return [0.0] * 384

        if self._fallback_mode:
            return self._fallback_embed(text)
            
        model = self._ensure_model()
        try:
            # encode returns numpy array, convert to list
            embedding = model.encode(text, convert_to_numpy=True).tolist()
            return embedding
        except Exception as e:
            logger.warning(f"Embedding generation failed: {str(e)}. Using deterministic hash embeddings.")
            self._fallback_mode = True
            return self._fallback_embed(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        if self._fallback_mode:
            return [self._fallback_embed(text) for text in texts]
            
        model = self._ensure_model()
        try:
            embeddings = model.encode(texts, convert_to_numpy=True).tolist()
            return embeddings
        except Exception as e:
            logger.warning(f"Batch embedding generation failed: {str(e)}. Using deterministic hash embeddings.")
            self._fallback_mode = True
            return [self._fallback_embed(text) for text in texts]

# --- DEPRECATED STUB ---
class _StubEmbedder(BaseEmbedder):
    """Original DummyEmbedder kept for reference but hidden"""
    def embed_text(self, text: str) -> List[float]:
        return [0.1] * 384
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [[0.1] * 384 for _ in texts]

# --- ACTIVATION ---
# Menggantikan DummyEmbedder dengan HuggingFaceEmbedder
# agar RetrievalPipeline langsung menggunakan implementasi nyata.
DummyEmbedder = HuggingFaceEmbedder
