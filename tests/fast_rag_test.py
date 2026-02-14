import sys
import json
import time
import logging
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.schemas.decision import LLMStrategy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

client = TestClient(app)

async def mock_generate(self, query, strategy, context=None):
    logger.info(f"⚡️ [MOCK LLM] Query: {query}")
    logger.info(f"⚡️ [MOCK LLM] Strategy: {strategy}")
    if context:
        logger.info(f"⚡️ [MOCK LLM] Context length: {len(context)}")
        logger.info(f"⚡️ [MOCK LLM] Context preview: {context[:100]}...")
    else:
        logger.info("⚡️ [MOCK LLM] No context provided.")
        
    if "hybrid" in query.lower():
        return "Mobil hybrid adalah kendaraan yang menggunakan dua jenis penggerak..."
    if "honda crv" in query.lower() and "spesifikasi" in query.lower():
        return "Spesifikasi mesin Honda CRV adalah 1.5L Turbo..."
    if "bandingkan" in query.lower():
        return "Honda CRV memiliki keunggulan di kenyamanan, sedangkan Toyota Fortuner tangguh di medan off-road..."
        
    return "Mocked response from LLM."

def run_fast_verification():
    print("🚀 STARTING FAST RAG VERIFICATION (MOCK LLM GENERATION)\n")
    print("ℹ️  Verifying Architecture Flow: Decision -> Retrieval -> Context -> LLM Provider")
    
    with patch("backend.app.llm.providers.hf_provider.HuggingFaceProvider.generate", side_effect=mock_generate, autospec=True) as mock_gen:
        
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
                response = client.post("/v1/chat/completions", json=case['payload'])
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
                
                passed = True
                reasons = []
                
                if "trace.use_rag" in case['expectations']:
                    if trace.get("use_rag") != case['expectations']["trace.use_rag"]:
                        passed = False
                        reasons.append(f"Expected RAG={case['expectations']['trace.use_rag']}, got {trace.get('use_rag')}")
                
                if "sources_count" in case['expectations']:
                    if len(sources) != case['expectations']['sources_count']:
                        passed = False
                        reasons.append(f"Expected sources={case['expectations']['sources_count']}, got {len(sources)}")
                
                if "sources_min" in case['expectations']:
                    if len(sources) < case['expectations']["sources_min"]:
                        passed = False
                        reasons.append(f"Expected min sources={case['expectations']['sources_min']}, got {len(sources)}")
                
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
            print("🎉 ALL ARCHITECTURE TESTS PASSED! System flow is verified.")
            sys.exit(0)
        else:
            print("⚠️ SOME TESTS FAILED.")
            sys.exit(1)

if __name__ == "__main__":
    run_fast_verification()
