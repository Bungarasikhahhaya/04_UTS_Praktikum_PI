`**Information Retrieval System (IR)**`
Program ini merupakan sistem Information Retrieval (IR) berbasis Python lokal yang dapat melakukan pencarian teks dari beberapa file dataset CSV (misalnya berita atau repository akademik).
Sistem ini menggunakan Whoosh untuk indexing, dan Cosine Similarity untuk melakukan ranking hasil pencarian.

**Persiapan & Instalasi**
1. Clone Repository atau Simpan File
Simpan tes.py dan folder dataset di satu direktori.

2. Install Semua Dependensi
Jalankan perintah berikut untuk menginstal library yang dibutuhkan:
**pip install pandas tqdm nltk whoosh scikit-learn**

**Cara Menjalankan Program**
Jalankan CLI:
**py tes.py**

**Alur Program**

**[1] – Load & Index Dataset**
- Fungsi: load_and_preprocess_batch()
- Input: Semua file .csv di folder ./dataset
- Proses:
  Membaca setiap CSV (seperti kompas.csv, tempo.csv, dll)
  Menggabungkan kolom judul dan konten
  Melakukan preprocessing:
  - Case folding → ubah ke huruf kecil
  - Hapus karakter non-huruf
  - Tokenisasi
  - Stopword removal
  - Membuat cache hasil preprocessing ke ./cache_preprocessing
  - Menghasilkan DataFrame gabungan seluruh dokumen
Dilanjutkan ke:
create_index(df, index_dir)
→ Membuat index Whoosh di folder ./indexdir.

**[2] – Search Query**
- Input Query dari pengguna
- Preprocessing Query
  Menggunakan fungsi preprocess_text() yang sama seperti dokumen.
  Membersihkan, tokenisasi, stopword removal.
- Whoosh Search (Tahap 1)
  Fungsi: search_and_rank()
  Sistem membuka indexdir
  Menggunakan MultifieldParser untuk mencari di field:
  - content
  - title
  Mengambil maksimal 20 hasil awal (retrieval tahap pertama)
- Cosine Similarity (Tahap 2 – Ranking Ulang)
  Sistem menghitung kemiripan antara query dan setiap dokumen hasil pencarian.
  Menggunakan CountVectorizer dan Cosine Similarity.
  Dokumen diurutkan dari skor tertinggi ke terendah.
- Output
  Ditampilkan 5 hasil teratas

**[3] – Count Docs per Dataset**
- Tujuan:
  Mengetahui berapa banyak data valid di setiap file dataset.
- Proses:
  - Membaca setiap file .csv di ./dataset
  - Mengecek kolom judul dan konten
  - Menghitung total baris dan baris valid (tidak kosong)

**[4] – Exit**
Keluar dari program
