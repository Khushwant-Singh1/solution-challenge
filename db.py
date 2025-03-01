import faiss

DB_FAISS_PATH = "vectorstores/db_faiss/index.faiss"

# Load the FAISS index
index = faiss.read_index(DB_FAISS_PATH)
print("FAISS index dimension:", index.d)
