from abc import ABC, abstractmethod
from typing import Optional
from backend.app.llm.strategies.base import BaseLLMStrategy

class BaseLLMProvider(ABC):
    """
    Abstract Base Class untuk semua LLM Provider.
    Memastikan setiap provider (DeepSeek, OpenAI, HF) mengikuti kontrak yang sama.
    """

    @abstractmethod
    async def generate(self, query: str, strategy: BaseLLMStrategy, context: Optional[str] = None) -> str:
        """
        Generate response dari LLM.
        Provider bertugas memanggil strategy.build_prompt() sebelum inferensi.
        """
        pass

    @abstractmethod
    async def stream(self, query: str, strategy: BaseLLMStrategy, context: Optional[str] = None):
        """
        Stream response generator.
        """
        pass
