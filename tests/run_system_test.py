import json
from statistics import mean

import requests


BASE_URL = "http://localhost:8000/v1/chat/completions"
REPORT_PATH = "tests/test_report.json"
REFUSAL_TEXT = "Maaf, sistem ini hanya mendukung pertanyaan seputar otomotif."

TEST_CASES = [
    {
        "name": "RAG HIT",
        "query": "Berapa kapasitas baterai Wuling Air EV?",
        "expected": {"use_rag": True, "min_score": 0.7},
    },
    {
        "name": "NO CONTEXT",
        "query": "Berapa kapasitas baterai Tesla Model Y?",
        "expected": {"use_rag": True, "context_length": 0},
    },
    {
        "name": "GUARDRAIL",
        "query": "Siapa presiden Indonesia?",
        "expected": {"refusal": True},
    },
]


def _require_keys(data: dict, keys: list[str]) -> str | None:
    for key in keys:
        if key not in data:
            return f"missing key '{key}' in response JSON"
    return None


def run_case(case: dict) -> dict:
    payload = {
        "query": case["query"],
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "use_rag": True,
    }

    detail = {
        "name": case["name"],
        "query": case["query"],
        "passed": False,
        "reason": "",
        "status_code": None,
        "latency_ms": 0.0,
        "retrieval_score": 0.0,
    }

    try:
        response = requests.post(BASE_URL, json=payload, timeout=180)
        detail["status_code"] = response.status_code

        if response.status_code != 200:
            detail["reason"] = f"status code {response.status_code}"
            return detail

        data = response.json()
        missing = _require_keys(data, ["answer", "sources", "trace"])
        if missing:
            detail["reason"] = missing
            return detail

        answer = data.get("answer", {}) or {}
        id_text = str(answer.get("id", "")).strip()
        en_text = str(answer.get("en", "")).strip()
        sources = data.get("sources", []) or []
        trace = data.get("trace", {}) or {}
        latency_ms = float(trace.get("latency_ms", 0.0) or 0.0)
        detail["latency_ms"] = latency_ms

        if not id_text or not en_text:
            detail["reason"] = "answer.id/en missing or empty"
            return detail

        expected = case["expected"]

        if expected.get("refusal"):
            if REFUSAL_TEXT not in id_text:
                detail["reason"] = "guardrail refusal text not found"
                return detail
            detail["passed"] = True
            return detail

        if expected.get("use_rag") and trace.get("use_rag") is not True:
            detail["reason"] = f"trace.use_rag expected True, got {trace.get('use_rag')}"
            return detail

        if "min_score" in expected:
            if not sources:
                detail["reason"] = "sources empty for RAG HIT"
                return detail
            top_score = float((sources[0] or {}).get("score", 0.0) or 0.0)
            detail["retrieval_score"] = top_score
            if top_score <= float(expected["min_score"]):
                detail["reason"] = f"top score <= {expected['min_score']} (got {top_score:.4f})"
                return detail

        if "context_length" in expected:
            context_length = int(trace.get("context_length", -1))
            if context_length != int(expected["context_length"]):
                detail["reason"] = (
                    "context_length mismatch, "
                    f"expected {expected['context_length']}, got {context_length}"
                )
                return detail
            if sources:
                top_score = float((sources[0] or {}).get("score", 0.0) or 0.0)
                detail["retrieval_score"] = top_score

        detail["passed"] = True
        return detail
    except Exception as exc:
        detail["reason"] = f"request failed: {exc}"
        return detail


def main() -> None:
    details = []

    for case in TEST_CASES:
        result = run_case(case)
        details.append(result)
        if result["passed"]:
            print(f"[PASS] {case['name']}")
        else:
            print(f"[FAIL] {case['name']} -> {result['reason']}")

    passed = len([d for d in details if d["passed"]])
    failed = len(details) - passed

    latencies = [d["latency_ms"] for d in details if d["latency_ms"] > 0]
    retrieval_scores = [d["retrieval_score"] for d in details if d["retrieval_score"] > 0]

    report = {
        "total": len(details),
        "passed": passed,
        "failed": failed,
        "average_latency_ms": float(mean(latencies)) if latencies else 0.0,
        "average_retrieval_score": float(mean(retrieval_scores)) if retrieval_scores else 0.0,
        "details": details,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nTOTAL: {passed}/{len(details)} PASSED")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
