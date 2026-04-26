import json

import requests


URL = "http://localhost:8000/v1/chat/completions"

TEST_CASES = [
    "What is an EV?",
    "Berapa kapasitas baterai Wuling Air EV?",
    "Siapa presiden Indonesia?",
]

for q in TEST_CASES:
    print("\n==============================")
    print("QUERY:", q)
    print("==============================")

    res = requests.post(
        URL,
        json={
            "query": q,
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        },
        timeout=180,
    )

    data = res.json()

    print("\n--- RESPONSE JSON ---")
    print(json.dumps(data, indent=2, ensure_ascii=False))

    print("\n--- GENERATED TEXT (ID) ---")
    print(data.get("answer", {}).get("id"))

    print("\n--- GENERATED TEXT (EN) ---")
    print(data.get("answer", {}).get("en"))
