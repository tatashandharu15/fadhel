from backend.app.schemas.decision import QueryType, RetrievalStrategy

class RagDecisionMaker:
    """
    Menentukan apakah RAG perlu diaktifkan dan strategi retrieval apa yang dipakai.
    """
    
    @staticmethod
    def decide(query_type: QueryType) -> tuple[bool, RetrievalStrategy]:
        """
        Returns: (use_rag, strategy)
        Logic deterministik berdasarkan QueryType.
        """
        if query_type == QueryType.DOMAIN_FACTUAL:
            return True, RetrievalStrategy.VECTOR_SIMILARITY
        elif query_type == QueryType.RECOMMENDATION:
            # Rekomendasi butuh keyword spesifik + semantic
            return True, RetrievalStrategy.HYBRID_KEYWORD
        else:
            # General knowledge tidak butuh RAG
            return False, RetrievalStrategy.NONE
