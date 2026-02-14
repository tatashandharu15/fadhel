from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field

class SourceType(str, Enum):
    MANUAL_PDF = "manual_pdf"
    SPEC_SHEET = "spec_sheet"
    WEB_ARTICLE = "web_article"
    INTERNAL_KB = "internal_kb"

class ContextMetadata(BaseModel):
    category: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    vehicle: Optional[str] = None
    year: Optional[int] = None

class ContextBlock(BaseModel):
    """
    Atomic unit of retrieved information.
    """
    source_id: str
    source_type: SourceType
    title: str
    content: str
    metadata: ContextMetadata

class ContextValidationStatus(str, Enum):
    VALID = "valid"
    INVALID = "invalid"
    PARTIAL = "partial" # Jika ada yang dibuang tapi masih ada sisa

class ContextResult(BaseModel):
    status: ContextValidationStatus
    valid_blocks: List[ContextBlock]
    errors: List[str] = []
    total_tokens: int = 0
