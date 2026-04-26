from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os

class ChatRequest(BaseModel):
    query: str
    model_id: str = Field(default_factory=lambda: os.getenv("DEFAULT_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct"))
    use_rag: bool = True
    filters: Optional[Dict] = None

class ChatResponse(BaseModel):
    answer: Dict[str, str]
    sources: List[Dict[str, Any]] = []
    trace: Optional[Dict[str, Any]] = None
