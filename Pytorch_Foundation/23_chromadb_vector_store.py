import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./chroma_db")

embeddings_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

collection = client.get_or_create_collection(name="customer_support_kb",embedding_function=embeddings_fn,metadata={"hnsw:space": "cosine"})

documents = [
    "I would like to get my money back for a faulty order.",
    "Where can I track the delivery status of my package?",
    "Can you help me update my account email address?",
    "Reimbursement process for canceled service charges."
]

metadatas = [
    {"category": "billing", "source_id": "KB-101"},
    {"category": "shipping", "source_id": "KB-102"},
    {"category": "account", "source_id": "KB-103"},
    {"category": "billing", "source_id": "KB-104"}
]

ids = ["doc_1", "doc_2", "doc_3", "doc_4"]

collection.add(documents=documents,metadatas=metadatas,ids=ids)

print(f"Total indexed documents in collection: {collection.count()}\n")

query_text = "How do I request a payment refund?"

results = collection.query(query_texts=query_text,n_results=2,include=["documents","metadatas","distances"])

print(f"Query: '{query_text}'\n")
print("--- ChromaDB Retrieval Results ---")

for doc , meta , dist in zip(results["documents"][0],results["metadatas"][0],results["distances"][0]):
    similarity = 1.0 - dist
    print(f"Score (Sim): {similarity:.4f} | Category: {meta['category']} | Text: {doc}")
