# Panduan Screenshot Backend – Automotive RAG Assistant

Dokumen ini menjelaskan 3 bagian backend yang paling relevan untuk di-screenshot, lengkap dengan:
- File dan path GitHub
- Function utama
- Alasan kenapa penting
- Elemen apa saja yang sebaiknya tampak di screenshot

Repository GitHub:  
https://github.com/tatashandharu15/fadhel

---

## 1. Orkestrasi Utama Sistem (End-to-End RAG + LLM)

- **File:** `backend/app/services/orchestrator.py`  
  GitHub:  
  https://github.com/tatashandharu15/fadhel/blob/main/backend/app/services/orchestrator.py
- **Class / Function:**  
  `class ChatOrchestrator` → `async def process_request(self, request: ChatRequest) -> ChatResponse`

**Kenapa penting:**
- Ini adalah service utama yang mengatur alur lengkap:
  - Menerima query dari API.
  - Memanggil `DecisionEngine` untuk menentukan apakah RAG dipakai dan strategi LLM apa.
  - Jika `use_rag = True`, memanggil `RetrievalPipeline` untuk mengambil dokumen dan menyusun context.
  - Memanggil provider LLM (Qwen) lewat `LLMFactory` dengan strategi yang sesuai.
  - Mengembalikan `ChatResponse` berisi jawaban, sources, latency, dan trace.
- Sangat representatif sebagai “fungsi utama sistem” untuk ditunjukkan di laporan/ujian.

**Yang sebaiknya terlihat di screenshot:**
- Signature fungsi:
  - `async def process_request(self, request: ChatRequest) -> ChatResponse:`
- Bagian “Step 1–3” di dalam fungsi:
  - Pemanggilan `DecisionEngine.analyze_request(request.query)`.
  - Blok `if decision.use_rag:` yang berisi:
    - `retrieved_docs = await self.retrieval_pipeline.run(request.query)`
    - Loop yang membangun `context_str` dan `sources`.
  - Pemanggilan LLM:
    - `llm_provider = LLMFactory.get_provider(request.model_id)`
    - `response_text = await llm_provider.generate(...)`
  - Penyusunan objek `ChatResponse(...)` dengan field `response`, `sources`, `latency_ms`, dan `trace`.

Dengan satu screenshot fungsi ini, penguji bisa melihat bahwa sistem benar-benar mengimplementasikan alur RAG + LLM end‑to‑end.

---

## 2. Implementasi Vector Store dan Cosine Similarity

- **File:** `backend/app/rag/retrieval/vector_store/faiss_store.py`  
  GitHub:  
  https://github.com/tatashandharu15/fadhel/blob/main/backend/app/rag/retrieval/vector_store/faiss_store.py
- **Class / Function:**  
  `class FaissVectorStore(BaseVectorStore)` → `add(...)` dan `search(...)`

**Kenapa penting:**
- Menunjukkan implementasi konkret dari **vector similarity search**:
  - Vektor embedding dokumen disimpan di `self.vectors_list`.
  - Query vector dinormalisasi dan dibandingkan dengan semua vektor dokumen menggunakan dot product.
  - Karena vektor sudah dinormalisasi, dot product merepresentasikan cosine similarity.
- Menghubungkan teori pada dokumen `metode_perhitungan.md` dengan implementasi nyata di kode.

**Yang sebaiknya terlihat di screenshot:**
- Deklarasi kelas:
  - `class FaissVectorStore(BaseVectorStore):`
- Fungsi `add(...)`:
  - Konversi list ke `np.array(...)`.
  - Pengecekan dimensi dan pemanggilan `_normalize(...)`.
  - Penyimpanan vektor dan metadata di dalam blok `with self._lock:`.
- Fungsi `search(...)`:
  - Normalisasi query vector.
  - Perhitungan skor dengan `np.dot(db_vectors, q_norm.flatten())`.
  - Pemilihan indeks skor tertinggi (`top_indices`) dan pembentukan hasil dengan `score` dan metadata.

Screenshot ini menunjukkan dengan jelas bahwa sistem benar‑benar melakukan pencarian berbasis cosine similarity di atas embedding yang disimpan.

---

## 3. Implementasi LLM Provider (Hugging Face)

- **File:** `backend/app/llm/providers/hf_provider.py`  
  GitHub:  
  https://github.com/tatashandharu15/fadhel/blob/main/backend/app/llm/providers/hf_provider.py
- **Class / Function:**  
  `class HuggingFaceProvider(BaseLLMProvider)` → `async def generate(self, query: str, strategy: BaseLLMStrategy, context: Optional[str] = None) -> str`

**Kenapa penting:**
- Menunjukkan bahwa sistem benar‑benar memanggil model generatif (Qwen) dari Hugging Face:
  - Lazy loading model via `_ensure_model_loaded()`.
  - Pembangunan prompt lewat `strategy.build_prompt(query, context)`.
  - Tokenisasi input dan pemanggilan `self.model.generate(...)` di CPU.
  - Decoding hasil menjadi teks jawaban.
- Relevan sebagai bukti “hasil uji coba berjalan” karena bagian ini adalah titik di mana inferensi model terjadi.

**Yang sebaiknya terlihat di screenshot:**
- Bagian `_ensure_model_loaded`:
  - Log `"[LLM] Initializing Hugging Face model: {self.model_id}"`.
  - Pemanggilan `AutoTokenizer.from_pretrained(self.model_id)` dan `AutoModelForCausalLM.from_pretrained(...)`.
- Bagian `generate(...)`:
  - `final_prompt = strategy.build_prompt(query, context)`
  - `inputs = self.tokenizer(final_prompt, return_tensors="pt")`
  - Blok `with torch.no_grad():` yang berisi pemanggilan `self.model.generate(...)` dengan `max_new_tokens`, `temperature`, `top_p`, dll.
  - Penghitungan `input_length` dan pemotongan token prompt sebelum decode.

Screenshot ini cocok dipasangkan dengan screenshot `ChatOrchestrator.process_request` untuk menunjukkan alur lengkap:
API → Orchestrator → RAG (FaissVectorStore) → HuggingFaceProvider → Jawaban.
