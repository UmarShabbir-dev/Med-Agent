import os

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from pathlib import Path


RAW_DOCS_DIR = "data/raw_docs"
VECTOR_DB_DIR = "data/vector_store"

def load_document(path):   # For to load documents from the folder
    all_docs = []
    for file in os.listdir(path):
        if file.endswith("pdf"):
            path = os.path.join(path,file)
            loader = PyPDFLoader(path)
            docs = loader.load()
            all_docs.extend(docs)    
    
    return all_docs


def chunk_document(docs): # Now next is to Chunk our documents into smaller sizes
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50, separators=["\n\n", "\n", ".", " "])
    return splitter.split_documents(docs)


def embed_and_store(chunks, persist_path): #Now embed docs and store to vectorDB
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/distiluse-base-multilingual-cased-v2")
    vectorstore = FAISS.from_documents(chunks,embeddings)
    vectorstore.save_local(persist_path)
    
if __name__ == "__main__":
    Path(VECTOR_DB_DIR).mkdir(parents=True, exist_ok=True)
    docs = load_document(RAW_DOCS_DIR)
    chunks = chunk_document(docs)
    embed_and_store(chunks, VECTOR_DB_DIR)