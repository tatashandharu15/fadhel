from pydantic import BaseModel

class ChunkingConfig(BaseModel):
    chunk_size: int = 400
    chunk_overlap: int = 80
    paragraph_aware: bool = True

class RetrievalConfig(BaseModel):
    top_k: int = 5
    similarity_threshold: float = 0.6 # Increased to filter irrelevant context
    hybrid_search_alpha: float = 0.5 # 0.5 = 50% keyword, 50% vector
