from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Fallback splitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langdetect import detect
from typing import List
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialDocumentProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        
    def load_documents(self, data_path: str = "./data") -> List:
        """Load financial documents with error handling"""
        try:
            loader = DirectoryLoader(
                data_path,
                glob="**/*.pdf",
                loader_cls=UnstructuredPDFLoader,
                loader_kwargs={"mode": "elements"},
                use_multithreading=True
            )
            return loader.load()
        except Exception as e:
            logger.error(f"Document loading failed: {str(e)}")
            raise

    def chunk_documents(self, documents: List, chunk_size: int = 1000, chunk_overlap: int = 200) -> List:
        """Split documents using reliable text splitter"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True
        )
        return text_splitter.split_documents(documents)

    def filter_vernacular(self, documents: List) -> List:
        """Basic language filtering for Indian context"""
        filtered = []
        for doc in documents:
            try:
                content = doc.page_content[:500]  # Check first 500 chars
                lang = detect(content)
                if lang in ["en", "hi", "ta"]:
                    filtered.append(doc)
            except:
                continue
        return filtered

    def build_knowledge_base(self) -> FAISS:
        """Create financial knowledge base"""
        documents = self.load_documents()
        filtered_docs = self.filter_vernacular(documents)
        chunks = self.chunk_documents(filtered_docs)
        
        return FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            normalize_L2=True
        )

if __name__ == "__main__":
    processor = FinancialDocumentProcessor()
    knowledge_base = processor.build_knowledge_base()
    knowledge_base.save_local("financial_db")
    
    # Test query
    results = knowledge_base.similarity_search("What is ELSS?", k=3)
    print(f"Found {len(results)} relevant documents")