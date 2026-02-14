from backend.app.services.decision.base import BaseQueryClassifier
from backend.app.schemas.decision import QueryType

class RegexQueryClassifier(BaseQueryClassifier):
    """
    Implementasi classifier deterministik sederhana berbasis aturan/regex.
    Bisa diganti dengan ML classifier di masa depan.
    """
    
    async def classify(self, query: str) -> QueryType:
        q = query.lower()
        
        # Rule 1: DOMAIN_FACTUAL (Fakta/Spesifikasi)
        factual_keywords = ["spesifikasi", "mesin", "kapasitas", "tahun", "cc", "harga", "kapan", "dimana", "berapa", "mobil", "fitur"]
        if any(k in q for k in factual_keywords):
            return QueryType.DOMAIN_FACTUAL
            
        # Rule 2: RECOMMENDATION (Analisis/Perbandingan)
        rec_keywords = ["bandingkan", "lebih baik", "vs", "rekomendasi"]
        if any(k in q for k in rec_keywords):
            return QueryType.RECOMMENDATION
            
        # Rule 3: GENERAL_KNOWLEDGE (Default)
        return QueryType.GENERAL_KNOWLEDGE
