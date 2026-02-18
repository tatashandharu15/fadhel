# Metode Perhitungan pada Sistem Automotive RAG Assistant

## 1. Pendahuluan

Sistem ini menggunakan pendekatan **Retrieval-Augmented Generation (RAG)**. Secara garis besar:
- Query pengguna diubah menjadi **vektor embedding**.
- Vektor query dibandingkan dengan vektor dokumen di **vektor store (FAISS)** menggunakan **vector similarity**.
- Dokumen paling relevan dikirim ke **LLM** sebagai konteks untuk menghasilkan jawaban akhir.

Dokumen ini menjelaskan secara akademik tiga komponen utama tersebut:
- Embedding
- Vector similarity
- Cosine similarity (termasuk contoh perhitungan manual).

---

## 2. Embedding

### 2.1 Definisi
**Embedding** adalah representasi numerik dari teks dalam bentuk vektor berdimensi tetap, misalnya \(\mathbb{R}^d\). Setiap kalimat atau dokumen diubah menjadi vektor:

\[
\text{teks} \rightarrow \mathbf{v} = [v_1, v_2, \dots, v_d]
\]

Tujuan embedding:
- Menangkap **makna semantik** teks.
- Memungkinkan komputasi kemiripan menggunakan operasi vektor (dot product, cosine similarity).

### 2.2 Implementasi di Sistem
Di dalam project ini:
- Modul embedding diimplementasikan melalui Sentence-Transformers (lihat keluarga file di `backend/app/rag/retrieval/embedder.py`).
- Hasil embedding disimpan ke vektor store FAISS (`backend/app/rag/retrieval/vector_store/faiss_store.py`).

---

## 3. Vector Similarity

### 3.1 Konsep
Setelah teks diubah menjadi vektor, kita perlu ukuran “seberapa mirip” dua vektor:
- Vektor query: \(\mathbf{q}\)
- Vektor dokumen: \(\mathbf{d}\)

**Vector similarity** adalah fungsi:
\[
\text{sim}(\mathbf{q}, \mathbf{d}) \rightarrow \mathbb{R}
\]

Semakin besar nilai similarity, semakin mirip query dan dokumen secara semantik. Berbagai metrik dapat digunakan:
- Dot product
- Cosine similarity
- Euclidean distance (umumnya diubah menjadi skor kedekatan).

### 3.2 Implementasi di Sistem
FAISS digunakan sebagai **vector store** untuk:
- Menyimpan embedding dokumen.
- Melakukan pencarian tetangga terdekat (nearest neighbors) berbasis similarity.

Secara konseptual:
1. Sistem menyiapkan matriks vektor dokumen \(D \in \mathbb{R}^{n \times d}\).
2. Untuk setiap query \(\mathbf{q}\), FAISS mencari dokumen dengan similarity tertinggi.

---

## 4. Cosine Similarity

### 4.1 Definisi Matematis
**Cosine similarity** mengukur sudut antara dua vektor:

\[
\text{cosine\_sim}(\mathbf{q}, \mathbf{d}) =
\frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\| \, \|\mathbf{d}\|}
\]

Dengan:
- \(\mathbf{q} \cdot \mathbf{d}\) = dot product antara dua vektor
- \(\|\mathbf{q}\|\) = norma (panjang) vektor \(\mathbf{q}\)
- \(\|\mathbf{d}\|\) = norma vektor \(\mathbf{d}\)

Nilai cosine similarity berada pada interval:
- \(-1 \leq \text{cosine\_sim} \leq 1\)
- Untuk embedding teks yang sudah dinormalisasi, nilai umumnya di rentang \([0, 1]\).

Interpretasi:
- Semakin mendekati **1** → vektor semakin searah → teks sangat mirip.
- Mendekati **0** → tidak berkorelasi (tidak mirip).
- Mendekati **-1** → berlawanan arah (jarang muncul dalam embedding teks umum).

---

## 5. Contoh Perhitungan Manual Cosine Similarity

### 5.1 Data
Misalkan:
- Query vector: \(\mathbf{q} = [0.2, 0.5, 0.1]\)
- Document vector: \(\mathbf{d} = [0.3, 0.4, 0.2]\)

Tujuan: hitung cosine similarity antara \(\mathbf{q}\) dan \(\mathbf{d}\).

### 5.2 Langkah 1 – Dot Product

\[
\mathbf{q} \cdot \mathbf{d} =
(0.2 \times 0.3) + (0.5 \times 0.4) + (0.1 \times 0.2)
\]
\[
= 0.06 + 0.20 + 0.02 = 0.28
\]

### 5.3 Langkah 2 – Norma Masing-masing Vektor

Norma (panjang) vektor didefinisikan sebagai:
\[
\|\mathbf{v}\| = \sqrt{v_1^2 + v_2^2 + \dots + v_d^2}
\]

#### 5.3.1 Norma Query \(\|\mathbf{q}\|\)
\[
\|\mathbf{q}\| = \sqrt{0.2^2 + 0.5^2 + 0.1^2}
= \sqrt{0.04 + 0.25 + 0.01}
= \sqrt{0.30}
\approx 0.5477
\]

#### 5.3.2 Norma Dokumen \(\|\mathbf{d}\|\)
\[
\|\mathbf{d}\| = \sqrt{0.3^2 + 0.4^2 + 0.2^2}
= \sqrt{0.09 + 0.16 + 0.04}
= \sqrt{0.29}
\approx 0.5385
\]

### 5.4 Langkah 3 – Cosine Similarity

Masukkan ke rumus:
\[
\text{cosine\_sim}(\mathbf{q}, \mathbf{d}) =
\frac{0.28}{0.5477 \times 0.5385}
\]

Hitung penyebut:
\[
0.5477 \times 0.5385 \approx 0.2945
\]

Sehingga:
\[
\text{cosine\_sim} \approx \frac{0.28}{0.2945} \approx 0.95
\]

### 5.5 Interpretasi Hasil

Nilai cosine similarity \(\approx 0.95\) sangat mendekati 1. Secara interpretasi:
- Vektor query dan vektor dokumen **sangat searah** di ruang vektor.
- Secara semantik, ini berarti **dokumen sangat relevan** dengan query.

Dalam konteks RAG dan pencarian dokumen:
- Dokumen dengan skor cosine similarity sekitar 0.95 akan berada di peringkat atas hasil pencarian.
- Sistem kemudian memilih beberapa dokumen dengan skor tertinggi untuk dijadikan **konteks** yang diberikan ke LLM, sehingga jawaban yang dihasilkan lebih akurat dan ter-grounded.

---

## 6. Kaitan dengan Implementasi Sistem

Ringkasnya:
- **Embedding**: Mengubah teks dokumen dan query menjadi vektor numerik menggunakan model embedding (Sentence-Transformers).
- **Vector similarity (Cosine)**: Digunakan oleh FAISS untuk mencari dokumen yang paling mirip dengan query.
- **RAG**: Dokumen yang ditemukan dikombinasikan dengan query dan diproses oleh LLM (Qwen/Qwen2.5-0.5B-Instruct) untuk menghasilkan jawaban bilingual yang sesuai dengan domain otomotif.

