from typing import List
from backend.app.rag.context.schema import ContextBlock

class ContextAssembler:
    """
    Assembles ContextBlocks into a string format ready for LLM System Prompt.
    """
    
    @staticmethod
    def assemble(blocks: List[ContextBlock]) -> str:
        if not blocks:
            return ""
            
        # Urutkan lagi untuk memastikan priority (just in case)
        sorted_blocks = sorted(blocks, key=lambda x: x.metadata.relevance_score, reverse=True)
        
        assembled_text = "Gunakan informasi konteks berikut untuk menjawab pertanyaan:\n\n"
        
        for i, block in enumerate(sorted_blocks, 1):
            assembled_text += f"--- SOURCE {i} ---\n"
            assembled_text += f"Title: {block.title}\n"
            assembled_text += f"Type: {block.source_type.value}\n"
            if block.metadata.vehicle:
                assembled_text += f"Vehicle: {block.metadata.vehicle} ({block.metadata.year or 'N/A'})\n"
            assembled_text += f"Content:\n{block.content}\n"
            assembled_text += "\n"
            
        return assembled_text
