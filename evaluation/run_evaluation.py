import csv
import json
import re
from functools import lru_cache
from pathlib import Path
from statistics import mean

import requests
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from sentence_transformers import SentenceTransformer, util

try:
    from rouge_score import rouge_scorer
except ImportError:
    class _SimpleRougeScore:
        def __init__(self, fmeasure: float):
            self.fmeasure = fmeasure


    class _FallbackRougeScorer:
        def __init__(self, metrics: list[str], use_stemmer: bool = True):
            self.metrics = metrics
            self.use_stemmer = use_stemmer

        @staticmethod
        def _tokenize(text: str) -> list[str]:
            return re.findall(r"\w+(?:[.,]\w+)?", text.lower())

        @staticmethod
        def _lcs_length(a: list[str], b: list[str]) -> int:
            if not a or not b:
                return 0
            dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
            for i in range(1, len(a) + 1):
                for j in range(1, len(b) + 1):
                    if a[i - 1] == b[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            return dp[-1][-1]

        def score(self, target: str, prediction: str) -> dict:
            target_tokens = self._tokenize(target)
            prediction_tokens = self._tokenize(prediction)
            lcs = self._lcs_length(target_tokens, prediction_tokens)
            precision = lcs / len(prediction_tokens) if prediction_tokens else 0.0
            recall = lcs / len(target_tokens) if target_tokens else 0.0
            fmeasure = (
                (2 * precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            return {"rougeL": _SimpleRougeScore(fmeasure)}


    class _FallbackRougeModule:
        RougeScorer = _FallbackRougeScorer


    rouge_scorer = _FallbackRougeModule()


API_URL = "http://localhost:8000/v1/chat/completions"
OUTPUT_PATH = Path("evaluation/results.json")
CSV_OUTPUT_PATH = Path("evaluation/results.csv")
SIMILARITY_THRESHOLD = 0.35
KEY_FACT_COVERAGE_THRESHOLD = 0.6

STOPWORDS = {
    "a",
    "adalah",
    "agar",
    "akan",
    "atau",
    "bagi",
    "bahwa",
    "banyak",
    "beberapa",
    "by",
    "dan",
    "dari",
    "dengan",
    "di",
    "for",
    "guna",
    "hingga",
    "itu",
    "its",
    "juga",
    "ke",
    "karena",
    "lebih",
    "many",
    "menggunakan",
    "merupakan",
    "of",
    "on",
    "pada",
    "paling",
    "para",
    "secara",
    "seperti",
    "serta",
    "suatu",
    "the",
    "to",
    "umumnya",
    "untuk",
    "yang",
}

EVAL_CASES = [
    {
        "query": "Berapa kapasitas baterai Wuling Air EV?",
        "reference": "Kapasitas baterai Wuling Air EV bervariasi berdasarkan varian, umumnya sekitar 17.3 kWh hingga 26.7 kWh.",
        "expected_keywords": ["wuling", "air", "ev", "17.3", "26.7", "kwh", "baterai"],
    },
    {
        "query": "Apa mesin yang digunakan Honda CR-V?",
        "reference": "Honda CR-V menggunakan mesin bensin 1.5L VTEC Turbo pada banyak varian, dan beberapa varian juga tersedia dalam konfigurasi hybrid e:HEV.",
        "expected_keywords": ["honda", "cr", "v", "1.5l", "vtec", "turbo", "hybrid", "ehev", "mesin"],
    },
    {
        "query": "Bandingkan Honda CR-V dan Toyota Fortuner.",
        "reference": "Honda CR-V cenderung unggul pada kenyamanan kabin dan efisiensi varian hybrid, sedangkan Toyota Fortuner kuat pada karakter SUV ladder frame dan kemampuan medan berat.",
        "expected_keywords": ["honda", "cr", "v", "toyota", "fortuner", "hybrid", "suv", "ladder", "frame"],
    },
    {
        "query": "Apa itu mobil listrik?",
        "reference": "Mobil listrik adalah kendaraan yang digerakkan motor listrik dengan sumber energi utama dari baterai yang dapat diisi ulang.",
        "expected_keywords": ["mobil", "listrik", "kendaraan", "motor", "baterai"],
    },
]


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


def _extract_answer_text(data: dict) -> str:
    answer = data.get("answer")
    if isinstance(answer, dict):
        id_text = str(answer.get("id", "")).strip()
        if id_text:
            return id_text
    return str(data.get("response", "")).strip()


def _extract_latency_ms(data: dict) -> float:
    trace = data.get("trace", {}) or {}
    if isinstance(trace, dict):
        trace_latency = trace.get("latency_ms")
        if trace_latency is not None:
            return float(trace_latency or 0.0)
    return float(data.get("latency_ms", 0.0) or 0.0)


@lru_cache(maxsize=1)
def _get_semantic_similarity_model() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[.,][0-9]+)?", text.lower())


def _normalize_keyword(token: str) -> str:
    return token.replace(",", ".").replace(":", "")


def _extract_keywords(text: str) -> set[str]:
    keywords = set()
    for token in _tokenize(text):
        normalized = _normalize_keyword(token)
        if normalized in STOPWORDS:
            continue
        if len(normalized) <= 2 and not any(ch.isdigit() for ch in normalized):
            continue
        keywords.add(normalized)
    return keywords


def _get_expected_keywords(case: dict) -> set[str]:
    manual_keywords = {_normalize_keyword(k.lower()) for k in case.get("expected_keywords", [])}
    if manual_keywords:
        return manual_keywords
    return _extract_keywords(case["reference"])


def _keyword_confusion(expected_keywords: set[str], generated_keywords: set[str]) -> tuple[int, int, int]:
    tp = len(expected_keywords & generated_keywords)
    fp = len(generated_keywords - expected_keywords)
    fn = len(expected_keywords - generated_keywords)
    return tp, fp, fn


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _contains_expected_key_facts(expected_keywords: set[str], generated_keywords: set[str]) -> bool:
    if not expected_keywords:
        return False
    coverage = _safe_ratio(len(expected_keywords & generated_keywords), len(expected_keywords))
    return coverage >= KEY_FACT_COVERAGE_THRESHOLD


def _calculate_semantic_similarity(reference: str, answer: str) -> float:
    if not reference.strip() or not answer.strip():
        return 0.0

    model = _get_semantic_similarity_model()
    embeddings = model.encode([reference, answer], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    return _clamp(similarity)


def _calculate_semantic_metrics(case: dict, answer: str, rouge_l: float, semantic_similarity: float) -> dict:
    expected_keywords = _get_expected_keywords(case)
    generated_keywords = _extract_keywords(answer)

    tp, fp, fn = _keyword_confusion(expected_keywords, generated_keywords)
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1_score = _safe_ratio(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0

    similarity = max(semantic_similarity, rouge_l, recall)
    is_correct = similarity >= SIMILARITY_THRESHOLD or _contains_expected_key_facts(expected_keywords, generated_keywords)
    accuracy = 1.0 if is_correct else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "expected_keywords": sorted(expected_keywords),
        "generated_keywords": sorted(generated_keywords),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def _calculate_confidence_score(retrieval_score: float, rouge_l: float, bleu: float) -> float:
    confidence = (0.5 * retrieval_score) + (0.25 * rouge_l) + (0.25 * bleu)
    return _clamp(confidence)


def _write_csv(rows: list[dict]) -> None:
    CSV_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query",
        "bleu",
        "rougeL",
        "semantic_similarity",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "confidence_score",
        "latency_ms",
        "retrieval_score",
    ]
    with CSV_OUTPUT_PATH.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def evaluate() -> dict:
    smoothie = SmoothingFunction().method1
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rows = []

    for case in EVAL_CASES:
        payload = {
            "query": case["query"],
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "use_rag": True,
        }
        row = {
            "query": case["query"],
            "reference": case["reference"],
            "response": "",
            "bleu": 0.0,
            "rougeL": 0.0,
            "semantic_similarity": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "confidence_score": 0.0,
            "latency_ms": 0.0,
            "retrieval_score": 0.0,
            "error": None,
        }
        try:
            resp = requests.post(API_URL, json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            answer = _extract_answer_text(data)
            sources = data.get("sources", []) or []
            latency_ms = _extract_latency_ms(data)
            retrieval_score = 0.0
            if sources and isinstance(sources[0], dict):
                retrieval_score = float(sources[0].get("score", 0.0) or 0.0)

            ref_tokens = case["reference"].split()
            ans_tokens = answer.split() if answer else []
            bleu = sentence_bleu([ref_tokens], ans_tokens, smoothing_function=smoothie) if ans_tokens else 0.0
            rouge_l = rouge.score(case["reference"], answer)["rougeL"].fmeasure if answer else 0.0
            semantic_similarity = _calculate_semantic_similarity(case["reference"], answer)
            semantic_metrics = _calculate_semantic_metrics(case, answer, rouge_l, semantic_similarity)
            confidence_score = _calculate_confidence_score(retrieval_score, rouge_l, bleu)

            row["response"] = answer
            row["bleu"] = float(bleu)
            row["rougeL"] = float(rouge_l)
            row["semantic_similarity"] = float(semantic_similarity)
            row["accuracy"] = semantic_metrics["accuracy"]
            row["precision"] = semantic_metrics["precision"]
            row["recall"] = semantic_metrics["recall"]
            row["f1_score"] = semantic_metrics["f1_score"]
            row["confidence_score"] = confidence_score
            row["latency_ms"] = latency_ms
            row["retrieval_score"] = retrieval_score
            row["expected_keywords"] = semantic_metrics["expected_keywords"]
            row["generated_keywords"] = semantic_metrics["generated_keywords"]
            row["tp"] = semantic_metrics["tp"]
            row["fp"] = semantic_metrics["fp"]
            row["fn"] = semantic_metrics["fn"]
        except Exception as exc:
            row["error"] = str(exc)
        rows.append(row)

    success = [r for r in rows if r["error"] is None]
    summary = {
        "avg_bleu": float(mean([r["bleu"] for r in success])) if success else 0.0,
        "avg_rougeL": float(mean([r["rougeL"] for r in success])) if success else 0.0,
        "avg_semantic_similarity": float(mean([r["semantic_similarity"] for r in success])) if success else 0.0,
        "avg_accuracy": float(mean([r["accuracy"] for r in success])) if success else 0.0,
        "avg_precision": float(mean([r["precision"] for r in success])) if success else 0.0,
        "avg_recall": float(mean([r["recall"] for r in success])) if success else 0.0,
        "avg_f1_score": float(mean([r["f1_score"] for r in success])) if success else 0.0,
        "avg_confidence_score": float(mean([r["confidence_score"] for r in success])) if success else 0.0,
        "avg_latency_ms": float(mean([r["latency_ms"] for r in success])) if success else 0.0,
        "avg_retrieval_score": float(mean([r["retrieval_score"] for r in success])) if success else 0.0,
    }
    result = {
        **summary,
        "total_queries": len(rows),
        "successful_queries": len(success),
        "failed_queries": len(rows) - len(success),
        "summary": summary,
        "details": rows,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    _write_csv(rows)
    return result


if __name__ == "__main__":
    metrics = evaluate()
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
