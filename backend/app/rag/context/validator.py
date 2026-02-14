from typing import List, Any
from backend.app.rag.context.schema import ContextResult, ContextValidationStatus, ContextBlock

class ContextValidator:
    """
    Enforces rules on the RAG context.
    """
    
    MAX_BLOCKS = 5
    MIN_SCORE_THRESHOLD = 0.5
    MAX_TOTAL_TOKENS = 2000 # Estimasi kasar
    
    @classmethod
    def validate(cls, blocks: List[ContextBlock]) -> ContextResult:
        valid_blocks = []
        errors = []
        seen_source_ids = set()
        current_tokens = 0
        
        # Sort by relevance score desc first
        sorted_blocks = sorted(blocks, key=lambda x: x.metadata.relevance_score, reverse=True)
        
        for block in sorted_blocks:
            # Rule 1: Content empty
            if not block.content.strip():
                errors.append(f"Block {block.source_id} ignored: Empty content")
                continue
                
            # Rule 2: Relevance Score
            if block.metadata.relevance_score < cls.MIN_SCORE_THRESHOLD:
                errors.append(f"Block {block.source_id} ignored: Score {block.metadata.relevance_score} < {cls.MIN_SCORE_THRESHOLD}")
                continue
            
            # Rule 3: Duplication
            if block.source_id in seen_source_ids:
                errors.append(f"Block {block.source_id} ignored: Duplicate source_id")
                continue
            
            # Rule 4: Max Blocks
            if len(valid_blocks) >= cls.MAX_BLOCKS:
                errors.append(f"Block {block.source_id} ignored: Max blocks limit ({cls.MAX_BLOCKS}) reached")
                continue
                
            # Rule 5: Token Limit (Estimasi: 1 char ~= 0.25 token, or just count chars/words for skeleton)
            # Simple estimation: words * 1.3
            estimated_tokens = len(block.content.split()) * 1.3
            if current_tokens + estimated_tokens > cls.MAX_TOTAL_TOKENS:
                 errors.append(f"Block {block.source_id} ignored: Token limit exceeded")
                 continue
            
            # If passed all checks
            seen_source_ids.add(block.source_id)
            valid_blocks.append(block)
            current_tokens += int(estimated_tokens)
            
        status = ContextValidationStatus.VALID
        if not valid_blocks and errors:
            status = ContextValidationStatus.INVALID
        elif errors:
             status = ContextValidationStatus.PARTIAL
             
        return ContextResult(
            status=status,
            valid_blocks=valid_blocks,
            errors=errors,
            total_tokens=current_tokens
        )
