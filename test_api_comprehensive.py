import requests
import json
import time
from datetime import datetime

tests = [
    {
        "name": "Test 1: Alternator Voltage (Domain Factual)",
        "query": "Berapa tegangan normal alternator saat mesin hidup?"
    },
    {
        "name": "Test 2: Wuling Battery Capacity (Specification)",
        "query": "Berapa kapasitas baterai Wuling Air EV varian Long Range?"
    },
    {
        "name": "Test 3: Comparison (CRV vs EV)",
        "query": "Apa perbedaan Honda CRV dengan mobil listrik?"
    },
    {
        "name": "Test 4: Troubleshooting",
        "query": "Kenapa lampu indikator aki selalu menyala?"
    },
    {
        "name": "Test 5: Out of Domain (Should Refuse)",
        "query": "Bagaimana cara membuat pizza yang enak?"
    }
]

url = "http://localhost:8000/v1/chat/completions"
print("\n" + "=" * 100)
print("🧪 RUNNING ALL TESTS".center(100))
print("=" * 100 + "\n")

for i, test in enumerate(tests, 1):
    print(f"\n{'=' * 100}")
    print(f"TEST {i}: {test['name']}")
    print(f"{'=' * 100}")
    print(f"Query: {test['query']}\n")
    
    try:
        payload = {
            "query": test['query'],
            "use_rag": True
        }
        
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=180)
        elapsed = (time.time() - start_time) * 1000
        
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ SUCCESS (HTTP {response.status_code}) - Response in {elapsed:.0f}ms\n")
        
        # Display answer
        print(f"📝 ANSWER (ID):")
        id_text = data['answer']['id']
        print(f"   {id_text[:150]}..." if len(id_text) > 150 else f"   {id_text}\n")
        
        print(f"📝 ANSWER (EN):")
        en_text = data['answer']['en']
        print(f"   {en_text[:150]}..." if len(en_text) > 150 else f"   {en_text}\n")
        
        # Display sources
        if data['sources']:
            print(f"📚 SOURCES: {len(data['sources'])} document(s) retrieved")
            for idx, src in enumerate(data['sources'][:2], 1):
                print(f"   {idx}. {src['payload']['title']}")
            if len(data['sources']) > 2:
                print(f"   ... and {len(data['sources'])-2} more")
        else:
            print("📚 SOURCES: None")
        
        # Display trace
        if data['trace']:
            trace = data['trace']
            print(f"\n📊 TRACE:")
            print(f"   Query Type: {trace.get('query_type', 'N/A')}")
            print(f"   LLM Strategy: {trace.get('llm_strategy', 'N/A')}")
            print(f"   Use RAG: {trace.get('use_rag', 'N/A')}")
            print(f"   Confidence: {trace.get('confidence', 'N/A')}")
            print(f"   Latency: {trace.get('latency_ms', 'N/A')}ms")
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}\n")

print("\n" + "=" * 100)
print("✅ ALL TESTS COMPLETED!".center(100))
print("=" * 100)
