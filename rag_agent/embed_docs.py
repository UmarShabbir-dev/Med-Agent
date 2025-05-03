import os

from langchain.document_loaders import PyPDFLoader


def load_document(path):   # For to load documents from the folder
    all_docs = []
    for file in os.listdir(path):
        if file.endswith("pdf"):
            path = os.path.join(path,file)
            loader = PyPDFLoader(path)
            docs = loader.load()
            all_docs.extend(docs)    
    
    return all_docs