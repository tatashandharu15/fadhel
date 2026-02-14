from backend.app.schemas.decision import DecisionResult, QueryType
from backend.app.services.decision.query_classifier import RegexQueryClassifier
from backend.app.services.decision.rag_decision import RagDecisionMaker
from backend.app.services.decision.llm_strategy import LLMStrategySelector

class DecisionEngine:
    """
    Facade utama untuk Decision Logic Layer.
    Menggabungkan Classifier, RAG Decision, dan Strategy Selector.
    """
    
    def __init__(self):
        # Dependency Injection bisa diterapkan di sini
        self.classifier = RegexQueryClassifier()
        self.rag_maker = RagDecisionMaker()
        self.strategy_selector = LLMStrategySelector()

    async def analyze_request(self, query: str) -> DecisionResult:
        """
        Flow utama decision making:
        1. Classify Query
        2. Decide RAG usage
        3. Select LLM Strategy
        4. Return Decision Object
        """
        
        # 1. Klasifikasi
        q_type = await self.classifier.classify(query)
        
        # 2. RAG Decision
        use_rag, ret_strategy = self.rag_maker.decide(q_type)
        
        # 3. LLM Strategy
        llm_strat = self.strategy_selector.select(q_type, use_rag)
        
        # 4. Construct Result
        
        # Calculate confidence score (Simplified)
        if q_type in [QueryType.DOMAIN_FACTUAL, QueryType.RECOMMENDATION]:
            confidence = 0.8
        else:
            confidence = 0.6
            
        # Generate simple explanation
        explanation = f"Classified as {q_type.value}. RAG: {'ON' if use_rag else 'OFF'}. Strategy: {llm_strat.value}."
        
        return DecisionResult(
            query_type=q_type,
            use_rag=use_rag,
            retrieval_strategy=ret_strategy,
            llm_strategy=llm_strat,
            confidence_score=confidence,
            explanation=explanation
        )
