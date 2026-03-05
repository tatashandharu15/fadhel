# How to Start and Setup

## 1) Prerequisites
- Docker
- Python 3.12
- Internet connection for first-time HuggingFace model download

## 2) Clone Repository
```bash
git clone https://github.com/tatashandharu15/fadhel.git
cd fadhel
```

## 3) Start Backend (FastAPI via Docker)
```bash
docker compose up -d --build
```

Backend API will be available at:
- `http://localhost:8000`
- Health check: `http://localhost:8000/health`

## 4) Run Streamlit UI
```bash
streamlit run client_ui/app.py
```

UI will be available at:
- `http://localhost:8501`

## 5) Optional Environment Configuration
- Default model is already set to:
  - `Qwen/Qwen2.5-0.5B-Instruct`
- To use private/rate-limited HuggingFace resources, set:
  - `HF_TOKEN` in your shell/runtime environment

Example:
```bash
export HF_TOKEN=your_hf_token_here
```

## 6) Quick API Test
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Apa kelebihan Wuling Air EV untuk penggunaan kota?",
    "model_id": "Qwen/Qwen2.5-0.5B-Instruct"
  }'
```

## 7) Notes
- First startup may take longer because embedding and model assets are downloaded.
- The system is automotive-only and will refuse non-automotive questions.
- RAG context is loaded from files in `data/`.
