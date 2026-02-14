import sys
import json
import time
import logging

# Configure logging to see backend logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from fastapi.testclient import TestClient
from backend.app.main import app

# Initialize TestClient
client = TestClient(app)

def run_e2e_tests():
    print("🚀 STARTING FULL E2E RAG VERIFICATION (REAL LLM & RETRIEVAL)\n")
    print("⚠️  Note: This test runs real inference (CPU). Expect latency.")
    
    test_cases = [
        {
            "name": "TEST CASE 1 — GENERAL KNOWLEDGE (RAG OFF)",
            "payload": {
                "query": "Apa itu mobil hybrid?",
                "model_id": "Qwen/Qwen2.5-0.5B-Instruct"
            },
            "expectations": {
                "trace.use_rag": False,
                "sources_count": 0
            }
        },
        {
            "name": "TEST CASE 2 — DOMAIN FACTUAL (RAG ON)",
            "payload": {
                "query": "Spesifikasi mesin Honda CRV",
                "model_id": "Qwen/Qwen2.5-0.5B-Instruct"
            },
            "expectations": {
                "trace.use_rag": True,
                "sources_min": 1,
                "keyword_in_response": "1.5L"
            }
        },
        {
            "name": "TEST CASE 3 — RECOMMENDATION (RAG ON + REASONING)",
            "payload": {
                "query": "Bandingkan Honda CRV dan Toyota Fortuner",
                "model_id": "Qwen/Qwen2.5-0.5B-Instruct"
            },
            "expectations": {
                "trace.use_rag": True,
                "sources_min": 1,
                "keyword_in_response": "Honda"
            }
        }
    ]
    
    results = []
    
    for case in test_cases:
        print(f"\n🧪 {case['name']}")
        print(f"Query: {case['payload']['query']}")
        start_t = time.time()
        
        try:
            # 60s timeout for CPU inference safety
            response = client.post("/v1/chat/completions", json=case['payload'], timeout=120.0)
            elapsed = time.time() - start_t
            
            if response.status_code != 200:
                print(f"❌ HTTP FAIL: {response.status_code}")
                print(response.text)
                results.append("FAIL")
                continue
                
            data = response.json()
            trace = data.get("trace", {})
            sources = data.get("sources", [])
            resp_text = data.get("response", "")
            
            print(f"⏱️  Latency: {elapsed:.2f}s")
            print(f"📋 Trace: RAG={trace.get('use_rag')}, Strat={trace.get('llm_strategy')}")
            print(f"📚 Sources: {len(sources)}")
            print(f"💬 Response Preview: {resp_text[:150]}...")
            
            # Verification
            passed = True
            reasons = []
            
            # Check RAG status
            if "trace.use_rag" in case['expectations']:
                if trace.get("use_rag") != case['expectations']["trace.use_rag"]:
                    passed = False
                    reasons.append(f"Expected RAG={case['expectations']['trace.use_rag']}, got {trace.get('use_rag')}")
            
            # Check Sources count
            if "sources_count" in case['expectations']:
                if len(sources) != case['expectations']['sources_count']:
                    passed = False
                    reasons.append(f"Expected sources={case['expectations']['sources_count']}, got {len(sources)}")
            
            if "sources_min" in case['expectations']:
                if len(sources) < case['expectations']["sources_min"]:
                    passed = False
                    reasons.append(f"Expected min sources={case['expectations']['sources_min']}, got {len(sources)}")
            
            # Check Content keywords
            if "keyword_in_response" in case['expectations']:
                kw = case['expectations']["keyword_in_response"]
                if kw.lower() not in resp_text.lower():
                    print(f"⚠️  Warning: Keyword '{kw}' not found in response.")
            
            if passed:
                print("✅ RESULT: PASS")
                results.append("PASS")
            else:
                print(f"❌ RESULT: FAIL ({', '.join(reasons)})")
                results.append("FAIL")
                
        except Exception as e:
            print(f"❌ EXCEPTION: {str(e)}")
            results.append("FAIL")
            
    print("\n--- SUMMARY ---")
    if all(r == "PASS" for r in results):
        print("🎉 ALL TESTS PASSED! RAG System is fully operational.")
        sys.exit(0)
    else:
        print("⚠️ SOME TESTS FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    run_e2e_tests()
