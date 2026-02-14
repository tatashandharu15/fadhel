from backend.app.llm.strategies.base import BaseLLMStrategy
from typing import Optional

class ContextAwareStrategy(BaseLLMStrategy):
    """
    Strategy for RAG. Combines context into prompt.
    Hardened Phase 3: Strict Automotive Domain, Context Priority & Safe Fallback.
    """
    
    def build_prompt(self, query: str, context: Optional[str] = None) -> str:
        # LOGIC SPLIT: RAG vs GENERAL
        if context and context.strip() and context != "NO_CONTEXT":
            return self._rag_prompt(query, context)
        else:
            return self._general_prompt(query)

    def _rag_prompt(self, query: str, context: str) -> str:
        return f"""<|im_start|>system
You are an Automotive Assistant.

CONTEXT:
{context}

TASK:
Answer using CONTEXT in Indonesian and English.
You MUST write at least 4 sentences for EACH language.

STRUCTURE:
1. Direct Answer based on Context.
2. Supporting detail from Context.
3. Another relevant detail or nuance.
4. Summary or final note from Context.

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

    def _general_prompt(self, query: str) -> str:
        return f"""<|im_start|>system
You are an Automotive Assistant.

TASK:
Answer in Indonesian and English.
You MUST write at least 4 sentences for EACH language.

STRUCTURE:
1. Definition/Main Answer.
2. Function/Usage.
3. Mechanism/How it works.
4. Additional Detail/Example.

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
