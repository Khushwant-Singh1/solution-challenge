from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from dotenv import load_dotenv
import tqdm
import time

load_dotenv()

DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'

def create_vector_db():
    try:
        # 1. Load documents
        print("🔄 Loading PDF documents...")
        loader = DirectoryLoader(DATA_PATH,
                               glob=['*.pdf'],
                               loader_cls=PyPDFLoader)
        documents = loader.load()
        print(f"✅ Successfully loaded {len(documents)} pages")

        # 2. Split text with smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Reduced from 1000
            chunk_overlap=100,
            length_function=len,
            add_start_index=True
        )
        texts = text_splitter.split_documents(documents)
        print(f"✂️ Split into {len(texts)} text chunks")

        # 3. Create embeddings with manual progress
        embeddings = OpenAIEmbeddings(
            model='text-embedding-3-large',
            request_timeout=60  # Increased timeout
        )

        # 4. Batch processing with progress
        print("🧠 Generating embeddings (this may take 5-15 minutes)...")
        start_time = time.time()
        
        # Process in batches to avoid timeouts
        batch_size = 100
        for i in tqdm.tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            if i == 0:  # Only create the initial DB
                db = FAISS.from_documents(batch, embeddings)
            else:       # Add subsequent batches
                db.add_documents(batch)
        
        # 5. Save and verify
        db.save_local(DB_FAISS_PATH)
        print(f"⏱️ Total processing time: {(time.time()-start_time)/60:.1f} minutes")
        print(f"💾 Saved to {DB_FAISS_PATH}")

        # Quick test
        test_query = "What is the psychology of money?"
        docs = db.similarity_search(test_query, k=1)
        if docs:
            print("\n🔍 Verification successful! Sample text:")
            print(docs[0].page_content[:200] + "...")
        else:
            print("\n⚠️ Verification failed - check your documents")

    except Exception as e:
        print(f"\n❌ Critical failure: {str(e)}")
        print("Possible solutions:")
        print("1. Reduce chunk_size to 500")
        print("2. Check PDFs are text-based (not scanned)")
        print("3. Verify OpenAI API key permissions")
        print("4. Monitor API usage at https://platform.openai.com/usage")

if __name__ == "__main__":
    create_vector_db()