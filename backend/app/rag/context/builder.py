from typing import List, Dict, Any
from backend.app.rag.context.schema import ContextBlock, ContextMetadata, SourceType

class ContextBuilder:
    """
    Transform raw retrieval results into structured ContextBlocks.
    """
    
    @staticmethod
    def build(raw_results: List[Dict[str, Any]]) -> List[ContextBlock]:
        blocks = []
        
        for res in raw_results:
            try:
                # Normalisasi data mentah ke schema
                # Asumsi raw_result punya struktur dict standar dari vector store wrapper
                
                meta = res.get("metadata", {})
                
                metadata = ContextMetadata(
                    category=meta.get("category", "general"),
                    relevance_score=res.get("score", 0.0),
                    vehicle=meta.get("vehicle"),
                    year=meta.get("year")
                )
                
                block = ContextBlock(
                    source_id=str(res.get("id", "unknown")),
                    source_type=SourceType(meta.get("type", "internal_kb")), # Default fallback
                    title=res.get("payload", {}).get("title", "Untitled"),
                    content=res.get("payload", {}).get("content", ""), # Raw content
                    metadata=metadata
                )
                
                # TODO: Implement summarization logic here if needed
                # block.content = summarize(block.content) 
                
                blocks.append(block)
                
            except Exception as e:
                # Log error transformation but don't crash entire build
                print(f"Error building block: {e}")
                continue
                
        return blocks
