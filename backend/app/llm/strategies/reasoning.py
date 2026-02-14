from backend.app.llm.strategies.base import BaseLLMStrategy
from typing import Optional

class ReasoningChainStrategy(BaseLLMStrategy):
    """
    Strategy for complex automotive reasoning (Comparisons, Analysis).
    Hardened Phase 3: Strict Domain, Reasoning Structure & Safety.
    """
    
    def build_prompt(self, query: str, context: Optional[str] = None) -> str:
        context_block = context if context else "NO_CONTEXT"
        
        return f"""<|im_start|>system
You are an Automotive Reasoning Assistant.

CONTEXT:
{context_block}

TASK:
Answer in Indonesian and English.
You MUST write at least 4 sentences for EACH language.

STRUCTURE:
1. Direct Comparison/Analysis.
2. Detail about Option A.
3. Detail about Option B.
4. Conclusion/Recommendation.

REFUSAL:
If NOT automotive, say:
[ID] Maaf, hanya otomotif.
[EN] Sorry, automotive only.

FORMAT:
[ID]
<Indonesian paragraph with 4 sentences>

[EN]
<English paragraph with 4 sentences>

User Query: {query}<|im_end|>
<|im_start|>assistant
[ID]
"""
