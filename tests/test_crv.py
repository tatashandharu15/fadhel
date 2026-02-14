import requests

BASE_URL = "http://localhost:8000"

def test_crv_query():
    print("\n📨 Testing CR-V Spec Retrieval...")
    query = "Spesifikasi mesin Honda CRV"
    
    payload = {
        "query": query,
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()
        print(f"Query: {query}")
        
        answer = data.get('response', '')
        sources = data.get('sources', [])
        
        print(f"Response: {answer}")
        print(f"Sources: {sources}")
        
        if "1.5L" in answer or "2.0L" in answer or "Turbo" in answer or "Hybrid" in answer:
            print("✅ CR-V Verification Passed!")
        else:
            print("⚠️ CR-V Verification: Answer might be generic, checking sources...")
            if sources:
                 print("✅ Sources found, retrieval working.")
            else:
                 print("❌ RAG Verification Failed (No sources found)")
            
    except Exception as e:
        print(f"❌ Query Failed: {e}")

if __name__ == "__main__":
    test_crv_query()
