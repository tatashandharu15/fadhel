from backend.app.services.decision.base import BaseQueryClassifier
from backend.app.schemas.decision import QueryType

class RegexQueryClassifier(BaseQueryClassifier):
    """
    Implementasi classifier deterministik sederhana berbasis aturan/regex.
    Bisa diganti dengan ML classifier di masa depan.
    """
    
    async def classify(self, query: str) -> QueryType:
        q = query.lower()

        auto_keywords = [
            "mobil",
            "kendaraan",
            "mesin",
            "cc",
            "baterai",
            "ev",
            "electric vehicle",
            "engine",
            "car",
            "vehicle",
            "hybrid",
            "listrik",
            "charging",
            "fuel",
        ]

        if any(k in q for k in auto_keywords):
            return QueryType.DOMAIN_FACTUAL

        return QueryType.GENERAL_KNOWLEDGE
