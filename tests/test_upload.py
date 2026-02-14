import requests
import sys

BASE_URL = "http://localhost:8000"

def test_upload():
    print("🚀 Testing Document Upload...")
    
    content = "Mobil terbang Esemka diprediksi akan meluncur pada tahun 2030. Mobil ini menggunakan tenaga fusi nuklir dingin dan mampu terbang hingga ketinggian 10.000 kaki."
    filename = "esemka_flying_car.txt"
    
    files = {
        'file': (filename, content, 'text/plain')
    }
    
    print(f"Uploading {filename}...")
    try:
        response = requests.post(f"{BASE_URL}/v1/documents/upload", files=files, timeout=60)
        response.raise_for_status()
        print(f"✅ Upload Success: {response.json()}")
    except Exception as e:
        print(f"❌ Upload Failed: {e}")
        if 'response' in locals():
            print(response.text)
        sys.exit(1)

def test_query():
    print("\n📨 Testing Retrieval...")
    query = "Kapan mobil terbang Esemka meluncur?"
    
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
        
        if "2030" in answer:
            print("✅ RAG Verification Passed!")
        else:
            print("❌ RAG Verification Failed (Context missing?)")
            
    except Exception as e:
        print(f"❌ Query Failed: {e}")

if __name__ == "__main__":
    test_upload()
    test_query()
