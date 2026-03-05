# System Overview — Automotive AI Assistant

| Kategori | Keterangan / Detail |
|---------|---------------------|
| Judul Penelitian | Automotive AI Assistant berbasis **Guardrail Domain Restriction** dan **Retrieval-Augmented Generation (RAG)** untuk tanya-jawab otomotif dengan sumber konteks terstruktur. |
| Tujuan Utama | (1) Menjawab pertanyaan seputar otomotif secara relevan, (2) memanfaatkan RAG agar jawaban berbasis konteks dokumen, (3) membatasi domain hanya ke automotive melalui aturan penolakan untuk pertanyaan di luar topik. |
| Dataset | Dataset lokal pada folder [data](file:///Users/tatas/Downloads/fadel/data): `wuling_air_ev.txt`, `honda_crv.txt`, `comparison_suv.md`. Data ini dipakai pada proses ingestion, chunking, embedding, lalu disimpan ke vector store untuk retrieval. |
| Model AI yang Digunakan | **Guardrail/Domain Restriction:** prompt & strategy rule untuk hanya menjawab topik otomotif. **Embedding Model:** HuggingFace Sentence-Transformers (digunakan pada pipeline retrieval). **LLM Generatif:** `Qwen/Qwen2.5-0.5B-Instruct` melalui provider HuggingFace. |
| Metodologi Utama | Pipeline utama: **Data Cleansing** (normalisasi konten dokumen) → **Chunking** (pecah dokumen menjadi unit retrieval) → **Embedding** (konversi chunk jadi vektor) → **Vector Database/Store (FAISS-style retrieval implementation)** → **Retrieval** (ambil top-k dokumen relevan) → **LLM Generation** (jawaban bilingual berbasis context + strategi). |
| Hasil Pengujian Performa | Metrik keluaran API menggunakan struktur respons backend: **`latency_ms`** (waktu proses end-to-end), **retrieval score** (nilai kemiripan/top result pada sumber), dan **response accuracy** (evaluasi deskriptif: jawaban sesuai konteks, konsisten dengan domain otomotif, serta kualitas penjelasan). |
| Evaluasi Sistem | Evaluasi dilakukan melalui: **API testing** endpoint chat/health, **RAG retrieval validation** (cek apakah sumber relevan muncul pada `sources` + skor), dan **domain restriction testing** (cek sistem menolak pertanyaan non-otomotif sesuai aturan guardrail). |
| Antarmuka Sistem | **Backend:** FastAPI (`/v1/chat/completions`, `/health`) untuk orkestrasi decision → retrieval → generation. **Frontend:** Streamlit untuk UI chat, state percakapan, pilihan bahasa, dan tampilan hasil/sumber. |
| Konfigurasi Perangkat Keras | Lingkungan pengembangan umum: CPU-based inference, Python 3.12, Docker Compose, memori RAM menengah (disarankan ≥8 GB untuk nyaman), cache model HuggingFace pada volume persisten, dan thread limit untuk stabilitas retrieval/inference. |

