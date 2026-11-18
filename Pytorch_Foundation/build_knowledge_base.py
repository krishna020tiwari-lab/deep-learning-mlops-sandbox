from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from kb_passages import PASSAGES, PASSAGE_CATEGORIES

DB_PATH = Path(__file__).resolve().parent / "chromadb"
client = chromadb.PersistentClient(path=str(DB_PATH))
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

try:
    client.delete_collection("kb_passages")
except Exception:
    pass

collection = client.create_collection(
    name="kb_passages",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"},
)

ids = [f"kb_{i}" for i in range(len(PASSAGES))]
metadatas = [{"category": c} for c in PASSAGE_CATEGORIES]
collection.upsert(ids=ids, documents=PASSAGES, metadatas=metadatas)
print("Stored:", collection.count())               # should print 30

res = collection.query(query_texts=["How do I request a payment refund?"], n_results=3,
                       include=["documents", "distances"])
for doc, dist in zip(res["documents"][0], res["distances"][0]):
    print(f"{1 - dist:.4f} | {doc}")