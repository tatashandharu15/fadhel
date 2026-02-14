import requests
import json

url = "http://localhost:8000/v1/chat/completions"
payload = {
    "query": "Apa itu turbo?",
    "use_rag": True
}
headers = {"Content-Type": "application/json"}

def main():
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        data = response.json()
        print("Response status:", response.status_code)
        print("Response body:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
