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
            "motor",
            "motorcycle",
            "sepeda motor",
            "kendaraan",
            "mesin",
            "cc",
            "aki",
            "accu",
            "alternator",
            "starter",
            "kelistrikan",
            "arus bocor",
            "tegangan",
            "soak",
            "tekor",
            "drop",
            "rem",
            "ban",
            "velg",
            "bearing",
            "rantai",
            "kopling",
            "transmisi",
            "karburator",
            "injektor",
            "knalpot",
            "shockbreaker",
            "suspensi",
            "overheat",
            "radiator",
            "coolant",
            "spooring",
            "balancing",
            "kampas",
            "cakram",
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