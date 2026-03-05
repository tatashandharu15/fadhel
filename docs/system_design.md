# System Design Documentation

## 1. Rancangan yang Sudah Dibuat
Project ini telah membangun **Automotive AI Assistant** sebagai sistem tanya jawab berbasis LLM yang fokus pada domain otomotif. Sistem menggunakan pendekatan **Retrieval-Augmented Generation (RAG)** agar jawaban tidak hanya mengandalkan parameter model, tetapi juga mengambil konteks dari knowledge base otomotif yang tersedia di project.  

Rancangan yang sudah terimplementasi mencakup:
- Sistem tanya jawab berbasis LLM untuk kebutuhan informasi otomotif.
- Integrasi RAG untuk mengambil konteks dokumen relevan sebelum proses generasi.
- Pemanfaatan dataset otomotif lokal sebagai knowledge source.
- Generasi jawaban bilingual **Indonesia (ID)** dan **English (EN)** sesuai kebijakan prompt/strategy.
- Guardrail domain agar sistem tetap fokus pada pertanyaan otomotif.

## 2. Produk yang Sudah Ada
Komponen yang sudah tersedia dalam project ini:

- **Backend API**
  - FastAPI sebagai service utama.
  - Endpoint utama: `/v1/chat/completions`
  - Endpoint pendukung: `/health` dan endpoint dokumen (upload/ingestion sesuai modul API).

- **LLM Engine**
  - Provider HuggingFace untuk inferensi model generatif.
  - Model utama: `Qwen/Qwen2.5-0.5B-Instruct`
  - Strategi LLM terpisah (direct answer, context-aware, reasoning chain).

- **Vector Database / Vector Store**
  - FAISS-style retrieval implementation pada layer vector store.
  - Penyimpanan embedding dokumen dan pencarian similarity untuk top-k context.

- **Knowledge Base**
  - Dataset tersimpan pada folder `data/` (contoh: `wuling_air_ev.txt`, `honda_crv.txt`, `comparison_suv.md`).
  - Diproses melalui pipeline ingestion + retrieval sebelum dipakai oleh LLM.

- **User Interface**
  - Streamlit chat interface.
  - Mendukung alur input pengguna, tampilan respons, sumber dokumen, dan state percakapan.

## 3. Rumusan Rancangan
### Masalah
Model LLM murni sering memberikan jawaban tanpa sumber eksplisit, berisiko kurang akurat untuk pertanyaan spesifik kendaraan, varian, dan spesifikasi teknis.

### Solusi
Sistem menerapkan **RAG** untuk mengambil dokumen relevan dari knowledge base otomotif, kemudian menyusun context sebelum dikirim ke LLM.

### Tujuan Sistem
- Memberikan jawaban yang lebih akurat dan ter-grounded.
- Membatasi domain jawaban hanya pada topik otomotif.
- Menyediakan sumber dokumen (`sources`) sebagai jejak referensi retrieval.

## 4. Tahap Tuning
Tahap tuning yang diterapkan pada sistem:

- **Prompt Hardening**
  - Penegasan domain restriction untuk automotive-only.
  - Instruksi penolakan tegas untuk pertanyaan non-automotive.

- **Prompt Engineering**
  - Format keluaran bilingual (ID/EN).
  - Struktur jawaban yang konsisten sesuai kebutuhan aplikasi.

- **Guardrail Rules**
  - Aturan bahwa pertanyaan di luar otomotif harus ditolak.
  - Kontrol agar model tidak mengklaim sumber yang tidak ada.

- **Pipeline Tuning**
  - Penyesuaian retrieval strategy agar context lebih relevan.
  - Context-aware generation untuk jawaban berbasis dokumen retrieval.

## 5. Penjelasan dan Skema Model
### LLM (Generative Model)
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Digunakan untuk:
  - Generate jawaban akhir.
  - Reasoning berbasis strategi.
  - Formatting output sesuai aturan sistem.

### Embedding Model
- Embedding berbasis HuggingFace/Sentence-Transformers.
- Digunakan untuk:
  - Mengubah teks dokumen dan query menjadi vektor numerik.

### Vector Database / Store
- FAISS-style vector retrieval.
- Digunakan untuk:
  - Similarity search antar vektor.
  - Pengambilan context paling relevan untuk RAG.

## 6. Skema Rancangan Sistem
Skema arsitektur sistem dalam bentuk diagram teks:

User  
↓  
Streamlit UI  
↓  
FastAPI Endpoint (`/v1/chat/completions`)  
↓  
Orchestrator  
↓  
Query Classification / Decision Engine  
↓  
RAG Retrieval (FAISS-style vector store)  
↓  
Context Builder  
↓  
LLM Generation (Qwen)  
↓  
Response (ID/EN + sources + trace)

## 7. Evaluasi
Metode evaluasi sistem yang digunakan:

- **Pengujian API**
  - Menggunakan request ke endpoint backend.
  - Memeriksa struktur response JSON (response, sources, latency_ms, trace).

- **Pengujian Retrieval**
  - Memastikan dokumen relevan berhasil ditemukan.
  - Memeriksa skor kemiripan/retrieval dan relevansi sumber.

- **Pengujian Guardrail**
  - Memastikan pertanyaan non-automotive ditolak sesuai aturan.
  - Memastikan sistem tetap fokus pada domain otomotif.

- **Pengujian UI**
  - Memastikan chat interface Streamlit berjalan normal.
  - Memastikan alur input → respons → tampilan sumber berjalan konsisten.

## 8. Perangkat Lunak yang Dipakai
- **Backend**
  - Python
  - FastAPI

- **AI Framework**
  - HuggingFace Transformers
  - Sentence Transformers

- **Vector Database / Retrieval**
  - FAISS (vector similarity search layer)

- **Frontend**
  - Streamlit

- **Containerization**
  - Docker / Docker Compose

