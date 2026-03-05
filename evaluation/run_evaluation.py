import json
from pathlib import Path
from statistics import mean

import requests
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer


API_URL = "http://localhost:8000/v1/chat/completions"
OUTPUT_PATH = Path("evaluation/results.json")

EVAL_CASES = [
    {
        "query": "Berapa kapasitas baterai Wuling Air EV?",
        "reference": "Kapasitas baterai Wuling Air EV bervariasi berdasarkan varian, umumnya sekitar 17.3 kWh hingga 26.7 kWh.",
    },
    {
        "query": "Apa mesin yang digunakan Honda CR-V?",
        "reference": "Honda CR-V menggunakan mesin bensin 1.5L VTEC Turbo pada banyak varian, dan beberapa varian juga tersedia dalam konfigurasi hybrid e:HEV.",
    },
    {
        "query": "Bandingkan Honda CR-V dan Toyota Fortuner.",
        "reference": "Honda CR-V cenderung unggul pada kenyamanan kabin dan efisiensi varian hybrid, sedangkan Toyota Fortuner kuat pada karakter SUV ladder frame dan kemampuan medan berat.",
    },
    {
        "query": "Apa itu mobil listrik?",
        "reference": "Mobil listrik adalah kendaraan yang digerakkan motor listrik dengan sumber energi utama dari baterai yang dapat diisi ulang.",
    },
]


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
            "latency_ms": 0.0,
            "retrieval_score": 0.0,
            "error": None,
        }
        try:
            resp = requests.post(API_URL, json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            answer = str(data.get("response", "")).strip()
            sources = data.get("sources", []) or []
            latency_ms = float(data.get("latency_ms", 0.0) or 0.0)
            retrieval_score = 0.0
            if sources and isinstance(sources[0], dict):
                retrieval_score = float(sources[0].get("score", 0.0) or 0.0)

            ref_tokens = case["reference"].split()
            ans_tokens = answer.split() if answer else []
            bleu = sentence_bleu([ref_tokens], ans_tokens, smoothing_function=smoothie) if ans_tokens else 0.0
            rouge_l = rouge.score(case["reference"], answer)["rougeL"].fmeasure if answer else 0.0

            row["response"] = answer
            row["bleu"] = float(bleu)
            row["rougeL"] = float(rouge_l)
            row["latency_ms"] = latency_ms
            row["retrieval_score"] = retrieval_score
        except Exception as exc:
            row["error"] = str(exc)
        rows.append(row)

    success = [r for r in rows if r["error"] is None]
    result = {
        "avg_bleu": float(mean([r["bleu"] for r in success])) if success else 0.0,
        "avg_rougeL": float(mean([r["rougeL"] for r in success])) if success else 0.0,
        "avg_latency_ms": float(mean([r["latency_ms"] for r in success])) if success else 0.0,
        "avg_retrieval_score": float(mean([r["retrieval_score"] for r in success])) if success else 0.0,
        "total_queries": len(rows),
        "successful_queries": len(success),
        "failed_queries": len(rows) - len(success),
        "details": rows,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    metrics = evaluate()
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
