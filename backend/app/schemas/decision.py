from enum import Enum
from pydantic import BaseModel
from typing import Optional, List

class QueryType(str, Enum):
    """Klasifikasi intent pengguna"""
    GENERAL_KNOWLEDGE = "general_knowledge"     # Konsep, edukasi, chit-chat
    DOMAIN_FACTUAL = "domain_factual"           # Spesifikasi, data teknis, manual
    RECOMMENDATION = "recommendation"           # Perbandingan, saran pembelian
    UNCERTAIN = "uncertain"                     # Tidak dapat diklasifikasikan

class LLMStrategy(str, Enum):
    """Strategi prompting dan inferensi"""
    DIRECT_ANSWER = "direct_answer"             # Jawab langsung (tanpa context berat)
    CONTEXT_AWARE = "context_aware"             # Jawab berdasarkan context RAG
    REASONING_CHAIN = "reasoning_chain"         # CoT (Chain of Thought) untuk analisis
    SUMMARIZATION = "summarization"             # Meringkas dokumen retrieval

class RetrievalStrategy(str, Enum):
    """Strategi pengambilan data (jika RAG aktif)"""
    NONE = "none"
    VECTOR_SIMILARITY = "vector_similarity"     # Standard semantic search
    HYBRID_KEYWORD = "hybrid_keyword"           # Semantic + Keyword match
    SPEC_LOOKUP = "spec_lookup"                 # Structured DB lookup (e.g., SQL/NoSQL)

class DecisionResult(BaseModel):
    """
    Output kontrak dari Decision Layer.
    Objek ini menentukan langkah selanjutnya bagi Orchestrator.
    """
    query_type: QueryType
    use_rag: bool
    retrieval_strategy: RetrievalStrategy
    llm_strategy: LLMStrategy
    confidence_score: float
    explanation: Optional[str] = None  # Untuk debugging/traceability
