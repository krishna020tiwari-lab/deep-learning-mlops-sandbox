import torch
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device {device}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

def get_sentence_embeddings(sentence):
    inputs = tokenizer(sentence,padding=True,truncation=True,return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * attention_mask,dim=1)
    sum_mask = torch.clamp(attention_mask.sum(dim=1),min=1e-9)
    sentence_embeddings = sum_embeddings / sum_mask
    return F.normalize(sentence_embeddings,p=2,dim=1)

query = "where is my parcel?"
documents = [
    "I would like to get my money back for a faulty order.",
    "Where can I track the delivery status of my package?",
    "Can you help me update my account email address?",
    "Reimbursement process for canceled service charges."
]

query_embedding = get_sentence_embeddings([query])
doc_embeddings = get_sentence_embeddings(documents)
print(f"Query Vector Shape:      {query_embedding.shape}")
print(f"Document Vectors Shape:  {doc_embeddings.shape}\n")
similarities = torch.mm(query_embedding,doc_embeddings.T).squeeze(0)
print(f"Similarities shape {similarities.shape}")
print(f"Query: '{query}'\n")
print("--- Similarity Search Results ---")

sorted_indices = torch.argsort(similarities,descending=True)
for idx in sorted_indices:
    score = similarities[idx].item()
    doc_text = documents[idx]
    print(f"Score: {score:.4f} | Document: '{doc_text}'")


