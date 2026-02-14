from abc import ABC, abstractmethod
from backend.app.schemas.decision import QueryType

class BaseQueryClassifier(ABC):
    """
    Interface untuk modul klasifikasi query.
    """
    
    @abstractmethod
    async def classify(self, query: str) -> QueryType:
        """
        Menganalisis query string dan mengembalikan kategori query.
        Implementasi bisa menggunakan Keyword Matching, Regex, atau Zero-shot classification.
        """
        pass
