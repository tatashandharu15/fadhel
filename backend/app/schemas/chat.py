from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os

class ChatRequest(BaseModel):
    query: str
    model_id: str = Field(default_factory=lambda: os.getenv("DEFAULT_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct"))
    use_rag: bool = True
    filters: Optional[Dict] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]] = []
    latency_ms: float = 0.0
    trace: Optional[Dict[str, Any]] = None # Added for debugging/tracing
