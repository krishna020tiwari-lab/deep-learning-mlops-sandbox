import chromadb
import torch
from chromadb.utils import embedding_functions
from transformers import AutoTokenizer,AutoModelForCausalLM
from pathlib import Path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

DB_PATH = Path(__file__).resolve().parent / "chroma_db"
client = chromadb.PersistentClient(path=str(DB_PATH))
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2",device="cuda")
collection = client.get_collection(name="kb_passages")

print("Items:",collection.count())
user_query = "How do i get a refund for a bad item?"
print(f"User Query: {user_query}")

retrieval_results = collection.query(query_texts=[user_query],n_results=3,include=["documents","metadatas","distances"])
MIN_SIM = 0.3
retrieved_docs = [d for d, dist in zip(retrieval_results["documents"][0], retrieval_results["distances"][0])
                  if 1 - dist >= MIN_SIM]
if not retrieved_docs:
    print("I don't know (no relevant passages found)")
    raise SystemExit
print("Retrieved Context Chunks ")

for i , doc in enumerate(retrieved_docs,1):
    print(f"Chunk {i} {doc}")

context_text = "\n".join([f"-  {doc}" for doc in retrieved_docs])

generator_model = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(generator_model)
model = AutoModelForCausalLM.from_pretrained(generator_model,dtype=torch.float16).to(device)

messages = [
    {
        "role": "system",
        "content": "You are a helpful support assistant. Answer the user's question strictly using ONLY the provided context. If the context does not contain the exact policy steps, say so clearly without inventing details."
    },
    {
        "role": "user",
        "content": f"Context:\n{context_text}\n\nQuestion: {user_query}"
    }
]

print("  Constructed ChatML Augmented Prompt  ")
formatted_prompt = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
print(formatted_prompt)
inputs = tokenizer.apply_chat_template(messages,add_generation_prompt=True,return_tensors="pt").to(device)

with torch.no_grad():
    output_ids = model.generate(**inputs,max_new_tokens=250,do_sample=False,pad_token_id=tokenizer.eos_token_id)

generated_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:],skip_special_tokens=True)

print("---- Final Grounded RAG Response ----")
print(generated_text)