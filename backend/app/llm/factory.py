from typing import Dict
from backend.app.llm.base import BaseLLMProvider
from backend.app.llm.providers.hf_provider import HuggingFaceProvider

class LLMFactory:
    """
    Factory untuk membuat instance LLM Provider.
    Mendukung singleton pattern per model_id jika perlu (caching).
    """
    
    _instances: Dict[str, BaseLLMProvider] = {}
    
    @classmethod
    def get_provider(cls, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct") -> BaseLLMProvider:
        """
        Mengembalikan instance provider untuk model_id tertentu.
        Default model: Qwen/Qwen2.5-0.5B-Instruct (CPU Friendly).
        """
        if model_id not in cls._instances:
            # Di sini bisa ada logic if/else jika support provider lain (e.g. OpenAI)
            # if "gpt" in model_id: return OpenAIProvider(model_id)
            
            cls._instances[model_id] = HuggingFaceProvider(model_id)
            
        return cls._instances[model_id]
