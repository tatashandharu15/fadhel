from __future__ import annotations

import hashlib
import logging
import re
import threading
from typing import List

import numpy as np

from backend.app.rag.retrieval.embedder import BaseEmbedder

logger = logging.getLogger(__name__)


class E5Embedder(BaseEmbedder):
    """
    Standalone embedder berbasis E5.

    Modul ini sengaja belum dihubungkan ke jalur utama RetrievalPipeline.
    Gunakan saat ingin mencoba model E5 secara terpisah.
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        self.model_name = model_name
        self._model = None
        self._fallback_mode = False
        self._lock = threading.Lock()

    def _fallback_embed(self, text: str) -> List[float]:
        vector = np.zeros(768, dtype=np.float32)
        tokens = re.findall(r"[\w-]+", text.lower())
        if not tokens:
            return vector.tolist()

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % 768
            vector[index] += 1.0

        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector.tolist()

    def _ensure_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        logger.info(f"Loading E5 embedding model: {self.model_name}...")
                        from sentence_transformers import SentenceTransformer

                        self._model = SentenceTransformer(self.model_name, device="cpu")
                        logger.info("E5 embedding model loaded successfully.")
                    except ImportError:
                        logger.warning(
                            "sentence-transformers library not found. Falling back to deterministic hash embeddings."
                        )
                        self._fallback_mode = True
                    except Exception as exc:
                        logger.warning(
                            f"Failed to load E5 model {self.model_name}: {exc}. Falling back to deterministic hash embeddings."
                        )
                        self._fallback_mode = True
        return self._model

    def _prepare_query(self, text: str) -> str:
        return f"query: {text.strip()}"

    def _prepare_passage(self, text: str) -> str:
        return f"passage: {text.strip()}"

    def embed_text(self, text: str) -> List[float]:
        if not text:
            return [0.0] * 768

        if self._fallback_mode:
            return self._fallback_embed(text)

        model = self._ensure_model()
        try:
            embedding = model.encode(
                self._prepare_query(text),
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).tolist()
            return embedding
        except Exception as exc:
            logger.warning(f"E5 query embedding failed: {exc}. Using deterministic hash embeddings.")
            self._fallback_mode = True
            return self._fallback_embed(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        if self._fallback_mode:
            return [self._fallback_embed(text) for text in texts]

        model = self._ensure_model()
        try:
            passages = [self._prepare_passage(text) for text in texts]
            embeddings = model.encode(
                passages,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).tolist()
            return embeddings
        except Exception as exc:
            logger.warning(f"E5 batch embedding failed: {exc}. Using deterministic hash embeddings.")
            self._fallback_mode = True
            return [self._fallback_embed(text) for text in texts]
