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

    @staticmethod
    def format_for_prompt(raw_results: List[Dict[str, Any]]) -> str:
        """
        Build context text that is easier for small models to consume.
        """
        if not raw_results:
            return ""

        sections: List[str] = []
        for res in raw_results:
            source_id = str(res.get("id", "unknown"))
            content = str(res.get("content", "")).strip()
            if not content:
                continue

            lines = [ln.strip(" -") for ln in content.splitlines() if ln.strip()]
            bullets = [f"- {ln}" for ln in lines if len(ln) > 2]
            if bullets:
                header = "Informasi kendaraan yang relevan:"
                if "wuling_air_ev" in source_id.lower():
                    header = "Mobil Wuling Air EV memiliki:"
                section = (
                    f"Sumber: {source_id}\n"
                    f"{header}\n"
                    + "\n".join(bullets[:12])
                )
                sections.append(section)

        return "\n\n".join(sections)
