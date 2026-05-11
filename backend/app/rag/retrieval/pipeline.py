from typing import List, Dict, Any, Optional
from backend.app.rag.retrieval.config import RetrievalConfig
from backend.app.rag.retrieval.embedder import BaseEmbedder, DummyEmbedder
from backend.app.rag.retrieval.retriever import BaseRetriever, InMemoryRetriever
from backend.app.rag.retrieval.chunker import TextChunker
from backend.app.rag.retrieval.reranker import DebertaV3Reranker

class RetrievalPipeline:
    """
    Orchestrator khusus untuk proses retrieval.
    Query -> Embed -> Retrieve -> Filter.
    Also handles Ingestion: Document -> Chunk -> Embed -> Store.
    """
    
    def __init__(self):
        # Dependency Injection (bisa diganti via config)
        self.config = RetrievalConfig()
        self.embedder: BaseEmbedder = DummyEmbedder()
        self.retriever: BaseRetriever = InMemoryRetriever()
        self.chunker = TextChunker()
        self.reranker = DebertaV3Reranker(self.config.reranker_model_id)
        
    async def ingest(self, text: str, metadata: Dict[str, Any]) -> int:
        """
        Ingest raw text into vector store.
        Returns number of chunks added.
        """
        # 1. Chunking
        chunks = self.chunker.chunk_document(text, metadata)
        
        if not chunks:
            return 0
            
        # 2. Embedding
        chunk_texts = [c["text"] for c in chunks]
        vectors = self.embedder.embed_batch(chunk_texts)
        
        # 3. Store
        # Prepare docs format for retriever
        docs = []
        for i, chunk in enumerate(chunks):
            doc = {
                "id": f"{metadata.get('filename', 'doc')}_{i}",
                "metadata": chunk["metadata"],
                "payload": {
                    "title": f"{metadata.get('filename', 'Untitled')} - Part {i+1}",
                    "content": chunk["text"]
                }
            }
            docs.append(doc)
            
        self.retriever.add_documents(vectors, docs)
        return len(docs)

    async def run(self, query: str, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Main entry point for retrieval operations.
        """
        
        # 1. Embed Query
        query_vector = self.embedder.embed_text(query)
        
        # 2. Retrieve Candidates
        raw_results = await self.retriever.retrieve(
            query_vector=query_vector, 
            config=self.config,
            filters=filters
        )

        # 3. Post-Retrieval Processing
        if self.config.enable_reranking and raw_results:
            raw_results = await self.reranker.rerank(query, raw_results)

        return raw_results
