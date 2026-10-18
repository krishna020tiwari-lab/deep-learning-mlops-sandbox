import torch
from transformers import AutoTokenizer,AutoModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

sentences = [
    "I deposited money in the bank",
    "We sat on the river bank"
]
inputs = tokenizer(sentences , padding=True, truncation=True, return_tensors="pt").to(device)

print("Input IDs Tensor Shape:", inputs["input_ids"].shape)
print("Attention Mask Shape:", inputs["attention_mask"].shape)

model.eval()
with torch.no_grad():
    outputs = model(**inputs)

last_hidden_state = outputs.last_hidden_state
print("Last Hidden State Shape" , last_hidden_state.shape)

tokens_sent1 = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
tokens_sent2 = tokenizer.convert_ids_to_tokens(inputs["input_ids"][1])

bank_idx_1 = tokens_sent1.index("bank")
bank_idx_2 = tokens_sent2.index("bank")

bank_vec1 = last_hidden_state[0, bank_idx_1]
bank_vec2 = last_hidden_state[1, bank_idx_2]

similarity = torch.cosine_similarity(bank_vec1.unsqueeze(0) , bank_vec2.unsqueeze(0))
print(f"\nCosine Similarity between 'bank' (financial) and 'bank' (river): {similarity.item():.4f}")