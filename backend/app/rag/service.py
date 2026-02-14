from typing import List, Dict, Any
from backend.app.schemas.decision import RetrievalStrategy
from backend.app.rag.retrieval.pipeline import RetrievalPipeline

class RagService:
    """
    Service facade for RAG operations.
    Menggunakan RetrievalPipeline yang sebenarnya (bukan stub logic di sini).
    """
    
    def __init__(self):
        self.pipeline = RetrievalPipeline()
        
    async def retrieve(self, query: str, strategy: RetrievalStrategy) -> List[Dict[str, Any]]:
        """
        Melakukan retrieval dokumen menggunakan Pipeline.
        """
        
        # Configure pipeline based on strategy (optional future feature)
        # if strategy == RetrievalStrategy.HYBRID_KEYWORD:
        #    self.pipeline.config.hybrid_search_alpha = 0.5
        
        print(f"[RagService] Retrieving for query: '{query}' with strategy: {strategy}")
        
        # Delegate to pipeline
        results = await self.pipeline.run(query)
        
        return results
