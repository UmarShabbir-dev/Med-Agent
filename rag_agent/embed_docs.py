import os

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_document(path):   # For to load documents from the folder
    all_docs = []
    for file in os.listdir(path):
        if file.endswith("pdf"):
            path = os.path.join(path,file)
            loader = PyPDFLoader(path)
            docs = loader.load()
            all_docs.extend(docs)    
    
    return all_docs

# Now next is to Chunk our documents into smaller sizes
def chunk_document(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50, separators=["\n\n", "\n", ".", " "])
    return splitter.split_documents(docs)