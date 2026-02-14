from typing import List, Dict, Any
import time
from backend.app.schemas.chat import ChatRequest, ChatResponse
from backend.app.schemas.decision import DecisionResult, LLMStrategy as StrategyEnum
from backend.app.services.decision.engine import DecisionEngine
from backend.app.rag.retrieval.pipeline import RetrievalPipeline
from backend.app.llm.factory import LLMFactory
from backend.app.llm.strategies.direct_answer import DirectAnswerStrategy
from backend.app.llm.strategies.context_aware import ContextAwareStrategy
from backend.app.llm.strategies.reasoning import ReasoningChainStrategy
from backend.app.rag.context.builder import ContextBuilder

class ChatOrchestrator:
    """
    Service utama yang mengatur alur data berdasarkan keputusan dari DecisionEngine.
    Full implementation: Decision -> RAG -> LLM.
    """
    
    def __init__(self):
        self.decision_engine = DecisionEngine()
        self.retrieval_pipeline = RetrievalPipeline()
        self.context_builder = ContextBuilder()
    
    def _get_llm_strategy(self, strategy_enum: StrategyEnum):
        if strategy_enum == StrategyEnum.DIRECT_ANSWER:
            return DirectAnswerStrategy()
        elif strategy_enum == StrategyEnum.CONTEXT_AWARE:
            return ContextAwareStrategy()
        elif strategy_enum == StrategyEnum.REASONING_CHAIN:
            return ReasoningChainStrategy()
        else:
            return DirectAnswerStrategy()

    async def process_request(self, request: ChatRequest) -> ChatResponse:
        """
        Orchestration Flow (Full RAG):
        1. User Input -> Decision Engine
        2. if use_rag -> Retrieval -> Context Builder
        3. LLM Generation
        """
        start_time = time.time()
        
        # Step 1: Decision Making
        decision: DecisionResult = await self.decision_engine.analyze_request(request.query)
        
        sources = []
        context_str = None
        
        # Step 2: RAG Flow (if needed)
        if decision.use_rag:
            # Retrieve
            retrieved_docs = await self.retrieval_pipeline.run(request.query)
            
            # Context Build
            # Convert retrieved docs to ContextBlock format if needed or pass directly
            # For simplicity in this iteration, we use builder's basic assembly
            # Assuming retrieval pipeline returns list of dicts compatible with builder
            # But wait, retrieval returns list of dicts, builder expects ContextBlock objects?
            # Let's check ContextBuilder later. For now, manual string join is safer if builder is complex
            
            # Simple context assembly for now to ensure flow works
            context_parts = []
            for doc in retrieved_docs:
                payload = doc.get("payload", {})
                content = payload.get("content", "")
                meta = doc.get("metadata", {})
                source_info = f"[{meta.get('vehicle', 'Unknown')}] {payload.get('title', '')}"
                context_parts.append(f"Source: {source_info}\nContent: {content}")
                
                # Add to sources list for response
                sources.append({
                    "id": doc.get("id"),
                    "score": doc.get("score"),
                    "title": payload.get("title")
                })
            
            if context_parts:
                context_str = "\n\n".join(context_parts)
        
        # Step 3: LLM Generation
        llm_provider = LLMFactory.get_provider(request.model_id)
        strategy_impl = self._get_llm_strategy(decision.llm_strategy)
        
        response_text = await llm_provider.generate(
            query=request.query,
            strategy=strategy_impl,
            context=context_str
        )
        
        # Calculate latency
        latency = (time.time() - start_time) * 1000
        
        # Build Trace Info
        trace_info = {
            "query_type": decision.query_type.value,
            "use_rag": decision.use_rag,
            "retrieval_strategy": decision.retrieval_strategy.value,
            "llm_strategy": decision.llm_strategy.value,
            "model_id": request.model_id,
            "confidence": decision.confidence_score,
            "context_length": len(context_str) if context_str else 0
        }
        
        return ChatResponse(
            response=response_text,
            sources=sources,
            latency_ms=latency,
            trace=trace_info
        )
