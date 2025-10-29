# main.py - Sistem IR CLI Tanpa Stemming (Mode Python Lokal)
# Disesuaikan untuk 5 CSV langsung di folder './dataset'

import os
import re
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import nltk
from nltk.corpus import stopwords
import shutil
from whoosh import index
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import MultifieldParser
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys

# --- KONFIGURASI LOKAL ---
# Asumsi 5 file CSV berada langsung di folder ini: ./dataset/file1.csv, ./dataset/file2.csv, dst.
DATASET_PATH = './dataset'
INDEX_DIR = 'indexdir'
CACHE_DIR = 'cache_preprocessing' 

# Inisialisasi Stopwords
try:
    # Memastikan paket nltk didownload
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        print("Mendownload nltk stopwords...")
        nltk.download('stopwords', quiet=True)
        
    STOPWORDS_ID = set(stopwords.words('indonesian'))
except Exception as e:
    print(f"Error inisialisasi NLTK: {e}")
    sys.exit(1)


# --- FUNGSI PREPROCESSING DAN LOAD DATA ---

def preprocess_text(text):
    """
    Melakukan Preprocessing: Case Folding, Menghilangkan non-huruf, Tokenization,
    dan Stopword Removal. (Tanpa Stemming).
    """
    # 1. Case Folding
    text = str(text).lower()
    # 2. Menghilangkan non-alfabet (kecuali spasi)
    text = re.sub(r'[^a-z\s]', ' ', text)
    # 3. Tokenization
    tokens = text.split()
    
    # 4. Stopword Removal dan filter token
    final_tokens = [t for t in tokens if t.isalpha() and t not in STOPWORDS_ID and len(t) > 2]

    return ' '.join(final_tokens)

def preprocess_row(args):
    """Fungsi pembantu untuk memproses satu baris data."""
    idx, row = args
    text = f"{row.get('judul', '')} {row.get('konten', '')}" 
    processed_text = preprocess_text(text)
    snippet = text[:200]
    
    return {'idx': idx, 'source': row['source'], 'filename': row['filename'],
            'content': processed_text, 'snippet': snippet}

def load_and_preprocess_batch(root_folder=DATASET_PATH, batch_size=5000, cache_prefix='cache_batch_'):
    """
    Memuat dataset dari file CSV yang ada langsung di root_folder, dan memproses secara batch.
    """
    if not os.path.exists(root_folder):
        print(f"❌ Error: Folder dataset tidak ditemukan di '{root_folder}'.")
        return None
        
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        
    # Hanya mencari file CSV langsung di root_folder, bukan di subfolder
    files_csv = [os.path.join(root_folder, f) for f in os.listdir(root_folder) if f.endswith('.csv')]
    
    data = []
    for file_path in files_csv:
        file_name = os.path.basename(file_path)
        source_name = file_name.replace('.csv', '') # Nama file (tanpa ekstensi) sebagai Source
        
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except:
            try:
                df = pd.read_csv(file_path, encoding='latin1')
            except Exception as e:
                print(f"⚠️ Gagal membaca file {file_path}: {e}")
                continue

        df.columns = [c.lower() for c in df.columns]
        
        if 'judul' in df.columns and 'konten' in df.columns:
            df['source'] = source_name
            # ID unik: Nama file_IndexBaris (misal: etd-usk_0)
            df['filename'] = [f"{source_name}_{i}" for i in range(len(df))] 
            data.append(df[['source', 'filename', 'judul', 'konten']].dropna(subset=['judul', 'konten']))
        
    if not data:
        print(f"⚠️ Tidak ditemukan dataset CSV yang valid di '{root_folder}'.")
        return None
        
    df_all = pd.concat(data, ignore_index=True)
    total_docs = len(df_all)
    print(f"Total dokumen yang akan diproses dari {len(files_csv)} file: {total_docs}")

    results_all = []

    # Proses per batch dengan Multiprocessing
    for start_idx in range(0, total_docs, batch_size):
        end_idx = min(start_idx + batch_size, total_docs)
        batch_file = os.path.join(CACHE_DIR, f"{cache_prefix}{start_idx}_{end_idx}.csv")
        
        if os.path.exists(batch_file):
            print(f"✅ Loading batch cache {start_idx} - {end_idx}")
            batch_df = pd.read_csv(batch_file)
        else:
            print(f"🔄 Processing batch {start_idx} - {end_idx} ({end_idx - start_idx} dokumen)...")
            
            pool = Pool(cpu_count())
            batch_rows = [(i, row) for i, row in df_all.iloc[start_idx:end_idx].iterrows()]
            
            results = list(tqdm(pool.imap(preprocess_row, batch_rows), total=len(batch_rows), desc=f"  Preprocessing batch {start_idx}"))
            
            pool.close()
            pool.join()
            
            results_sorted = sorted(results, key=lambda x: x['idx'])
            batch_df = pd.DataFrame([{
                'source': r['source'],
                'filename': r['filename'],
                'content': r['content'],
                'snippet': r['snippet']
            } for r in results_sorted])
            
            batch_df.to_csv(batch_file, index=False)
            print(f"  Batch {start_idx} - {end_idx} disimpan ke cache.")
            
        results_all.append(batch_df)

    df_full = pd.concat(results_all, ignore_index=True)
    print(f"\nSemua batch selesai, total dokumen terproses: {len(df_full)}")
    return df_full


# --- FUNGSI INDEXING DAN PENCARIAN/RANKING ---

def create_index(df, index_dir=INDEX_DIR, force=True):
    """Membuat index Whoosh baru."""
    if force and os.path.exists(index_dir):
        print(f"⚠️ Menghapus index lama di {index_dir}")
        shutil.rmtree(index_dir)
        
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)
        
    print(f"🔄 Memulai indexing {len(df)} dokumen...")
    
    schema = Schema(
        title=ID(stored=True, unique=True),
        path=ID(stored=True),
        content=TEXT(stored=True),
        snippet=TEXT(stored=True)
    )
    
    try:
        ix = index.create_in(index_dir, schema)
        writer = ix.writer()

        for _, row in tqdm(df.iterrows(), total=len(df), desc="  Indexing dokumen"):
            writer.add_document(
                title=row['filename'],
                path=row['source'],
                content=row['content'],
                snippet=row['snippet']
            )
            
        writer.commit(optimize=True)
        print("✅ Indexing selesai.\n")
    except Exception as e:
        print(f"❌ Gagal membuat index: {e}")
        
def search_and_rank(query_str, index_dir=INDEX_DIR, top_k=5):
    """
    Melakukan pencarian (Whoosh) dan ranking ulang dengan
    BoW CountVectorizer dan Cosine Similarity.
    """
    if not os.path.exists(index_dir) or not os.listdir(index_dir):
        print("⚠️ Index belum tersedia! Silakan load dan index terlebih dahulu.")
        return
        
    print("🔄 Melakukan pencarian dan ranking...")
    
    try:
        ix = index.open_dir(index_dir)
    except:
        print("❌ Index tidak dapat dibuka. Coba re-index.")
        return
        
    query_preproc = preprocess_text(query_str)
    
    # Whoosh Search (Initial Retrieval)
    parser = MultifieldParser(['content', 'title'], schema=ix.schema)
    query = parser.parse(query_preproc)
    
    with ix.searcher() as searcher:
        hits = searcher.search(query, limit=20) 
        results = []
        for hit in hits:
            results.append({
                'title': hit['title'],
                'path': hit['path'],
                'content': hit['content'], 
                'snippet': hit['snippet']
            })
            
    if not results:
        print("⚠️ Tidak ada hasil ditemukan.")
        return

    # VSM Ranking (Re-ranking) menggunakan Cosine Similarity
    doc_texts = [r['content'] for r in results]
    
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform([query_preproc] + doc_texts)
    
    cosine_sim = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    ranked = sorted(zip(results, cosine_sim), key=lambda x: x[1], reverse=True)[:top_k]
    
    print(f"\n=== {top_k} HASIL TERATAS DENGAN COSINE SIMILARITY ===")
    for i, (res, score) in enumerate(ranked, 1):
        print(f"{i}. {res['title']} ({res['path']}) — Skor: {score:.4f}")
        print(f"   Snippet: {res['snippet']}...\n")
        
def count_docs_per_file(root_folder=DATASET_PATH):
    """Menghitung jumlah dokumen per file CSV di root_folder."""
    if not os.path.exists(root_folder):
        print(f"❌ Error: Folder dataset tidak ditemukan di '{root_folder}'.")
        return
        
    files_csv = [os.path.join(root_folder, f) for f in os.listdir(root_folder) if f.endswith('.csv')]
    
    print(f"Mencari {len(files_csv)} file di {root_folder}...")
    for file_path in files_csv:
        file_name = os.path.basename(file_path)
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except:
            df = pd.read_csv(file_path, encoding='latin1')

        df.columns = [c.lower() for c in df.columns]
        total_docs = len(df)
        
        if 'judul' in df.columns and 'konten' in df.columns:
            valid_docs = df[['judul', 'konten']].dropna().shape[0]
        else:
            valid_docs = 0
            
        print(f"'{file_name}': Total = {total_docs}, Baris valid = {valid_docs}")


# --- FUNGSI UTAMA (MAIN CLI) ---

def main():
    """Fungsi utama untuk menjalankan CLI sistem IR."""
    
    if os.path.exists(INDEX_DIR) and os.listdir(INDEX_DIR):
        print("ℹ️ Index directory ditemukan. Sistem siap untuk pencarian.")

    while True:
        print("""
====================================
=== INFORMATION RETRIEVAL SYSTEM ===
====================================
[1] Load & Index Dataset
[2] Search Query
[3] Count Docs per Dataset
[4] Exit
====================================
        """)
        
        choice = input("Pilih menu: ")
        
        if choice == '1':
            print("▶️ Memulai load, preprocess, dan indexing ...")
            df_indexed = load_and_preprocess_batch(root_folder=DATASET_PATH) 
            if df_indexed is not None and not df_indexed.empty:
                create_index(df_indexed, index_dir=INDEX_DIR, force=True) 
                
        elif choice == '2':
            query = input("Masukkan query pencarian: ")
            if query.strip():
                search_and_rank(query, index_dir=INDEX_DIR, top_k=5)
            else:
                print("❌ Query tidak boleh kosong.")
                
        elif choice == '3':
            count_docs_per_file(root_folder=DATASET_PATH)
            
        elif choice == '4':
            print("👋 Terima kasih sudah menggunakan sistem IR!")
            break
            
        else:
            print("❌ Pilihan tidak valid!")

if __name__ == "__main__":
    main()