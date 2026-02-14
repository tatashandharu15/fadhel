import requests
import time
import json

def test_chat():
    url = "http://localhost:8000/v1/chat/completions"
    print(f"🚀 Testing API at {url}")
    
    # 1. Health Check
    try:
        print("Checking health...")
        health = requests.get("http://localhost:8000/health", timeout=30)
        print(f"✅ Health Check: {health.status_code} {health.json()}")
    except Exception as e:
        print(f"❌ Health Check Failed: {e}")
        return

    # 2. Chat Request
    payload = {
        "query": "Spesifikasi mesin Honda CRV",
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct"
    }
    
    print(f"\n📨 Sending Chat Request (this triggers the real LLM)...")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    start = time.time()
    try:
        response = requests.post(url, json=payload, timeout=300) # 5 min timeout
        elapsed = time.time() - start
        
        print(f"\n⏱️ Time taken: {elapsed:.2f}s")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Response received:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            print("❌ Error response:")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_chat()
