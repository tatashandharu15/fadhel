import os

from pydantic import BaseModel, Field

class ChunkingConfig(BaseModel):
    chunk_size: int = 400
    chunk_overlap: int = 80
    paragraph_aware: bool = True

class RetrievalConfig(BaseModel):
    top_k: int = 5
    similarity_threshold: float = 0.25 # Keep recall high; reranking/filtering can narrow later
    hybrid_search_alpha: float = 0.5 # 0.5 = 50% keyword, 50% vector
    enable_reranking: bool = True
    reranker_model_id: str = Field(
        default_factory=lambda: os.getenv(
            "RERANKER_MODEL_ID",
            "cross-encoder/ms-marco-deberta-v3-base",
        )
    )
