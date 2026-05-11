from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DebertaV3Reranker:
    """
    Reranker berbasis DeBERTa v3 untuk mengurutkan kandidat retrieval.

    Default model diarahkan ke cross-encoder DeBERTa v3 yang memang dilatih
    untuk scoring query-document pair.
    """

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.device = DEVICE
        self._model_loaded = False
        self._load_failed = False
        self._lock = threading.Lock()
        self.tokenizer = None
        self.model = None

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not documents or self._load_failed:
            return documents

        return await asyncio.to_thread(self._rerank_sync, query, documents)

    def _ensure_model_loaded(self) -> bool:
        if self._model_loaded:
            return True

        with self._lock:
            if self._model_loaded:
                return True

            try:
                logger.info(f"[RERANK] Loading DeBERTa v3 reranker: {self.model_id}")
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                model_dtype = torch.float16 if self.device == "cuda" else torch.float32
                load_kwargs = {
                    "torch_dtype": model_dtype,
                }
                if self.device == "cuda":
                    load_kwargs["device_map"] = "auto"

                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_id,
                    **load_kwargs,
                )
                self.model.eval()
                self._model_loaded = True
                logger.info(f"[RERANK] Reranker {self.model_id} loaded successfully.")
                return True
            except Exception as exc:
                self._load_failed = True
                logger.warning(f"[RERANK] Failed to load reranker {self.model_id}: {exc}")
                return False

    def _score_documents(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self._ensure_model_loaded():
            return documents

        pairs = [(query, str(doc.get("content", ""))) for doc in documents]
        if not pairs:
            return documents

        encoded = self.tokenizer(
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        if self.device == "cuda":
            encoded = {key: value.to(next(self.model.parameters()).device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)
            logits = outputs.logits

        if logits.ndim == 1:
            raw_scores = logits
        elif logits.shape[-1] == 1:
            raw_scores = logits.squeeze(-1)
        else:
            raw_scores = logits[:, -1]

        if self.device == "cuda":
            torch.cuda.empty_cache()

        scored_documents = []
        for document, raw_score in zip(documents, raw_scores):
            score_value = torch.sigmoid(raw_score).item() if torch.is_tensor(raw_score) else float(raw_score)
            ranked_document = document.copy()
            ranked_document["vector_score"] = float(document.get("score", 0.0) or 0.0)
            ranked_document["score"] = float(score_value)
            scored_documents.append(ranked_document)

        scored_documents.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return scored_documents

    def _rerank_sync(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            return self._score_documents(query, documents)
        except Exception as exc:
            logger.warning(f"[RERANK] Falling back to retrieval order: {exc}")
            return documents