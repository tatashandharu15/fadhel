from abc import ABC, abstractmethod
from typing import Optional

class BaseLLMStrategy(ABC):
    """
    Interface untuk strategi penyusunan prompt.
    Tugasnya HANYA membentuk prompt final berdasarkan input,
    tanpa melakukan inferensi.
    """
    
    @abstractmethod
    def build_prompt(self, query: str, context: Optional[str] = None) -> str:
        """
        Menyusun prompt final.
        """
        pass
