from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

from transformers import pipeline

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/distiluse-base-multilingual-cased-v2")
vector_store = FAISS.load_local("data/vector_store", embedding_model,allow_dangerous_deserialization = True)

llm_pipeline = pipeline("text-generation", model="gpt2")

print(" Ask a medical question (English or German). Type 'exit' to quit.")

while True:
    q = input("\n> ")
    if q.strip().lower() in ["exit","quit"]:
        break
        
    docs = vector_store.similarity_search(q,k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"Frage: {q}\n\nKontext:\n{context}\n\nAntwort:"

    result = llm_pipeline(prompt, max_new_tokens=100, do_sample=True)[0]["generated_text"]
    print("\n🧠 Antwort:\n", result.replace(prompt, "").strip())
    
    