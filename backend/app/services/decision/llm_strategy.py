from backend.app.schemas.decision import QueryType, LLMStrategy

class LLMStrategySelector:
    """
    Menentukan bagaimana LLM harus diprompt/dikonfigurasi.
    """
    
    @staticmethod
    def select(query_type: QueryType, use_rag: bool) -> LLMStrategy:
        """
        Memilih strategi inferensi.
        """
        if query_type == QueryType.DOMAIN_FACTUAL:
            return LLMStrategy.CONTEXT_AWARE
        elif query_type == QueryType.RECOMMENDATION:
            return LLMStrategy.REASONING_CHAIN
        else:
            return LLMStrategy.DIRECT_ANSWER
