import torch
import torch.nn as nn
from  torch.utils.data import DataLoader,Dataset
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from torch.optim import AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device {device}")

train_texts = [
    "I want to cancel my subscription immediately.",
    "Can I get a refund for my recent purchase?",
    "How do I update my credit card details?",
    "Where is my package? It is delayed.",
    "Please delete my account and personal data.",
    "I was charged twice on my billing statement.",
    "Track my shipment status for order 8842.",
    "Change my delivery shipping address."]

train_labels = [1, 0, 0, 2, 1, 0, 2, 2]

class CustomIntentDataset(Dataset):
    def __init__(self,texts,labels,tokenizer,max_length=32):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        return {"input_ids":encoding["input_ids"].squeeze(0),"attention_mask":encoding["attention_mask"].squeeze(0),"labels": torch.tensor(self.labels[idx], dtype=torch.long)}

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=3).to(device)
dataset = CustomIntentDataset(train_texts, train_labels, tokenizer)
dataloader = DataLoader(dataset,batch_size=2,shuffle=True)

optimizer = AdamW(model.parameters(),lr=2e-5)
epochs = 4
model.train()
print("\n Starting bert fine tuning")
for epoch in range(epochs):
    total_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids,attention_mask=attention_mask,labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{epochs}] | Loss: {avg_loss:.4f}")

model.eval()
test_sentences = [ "I need money back for my failed order.", "Where is my item right now?" ]
label_map = {0: "Billing/Refunds", 1: "Account Management", 2: "Shipping/Tracking"}
with torch.no_grad():
    for text in test_sentences:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        outputs = model(**inputs)
        pred_idx = torch.argmax(outputs.logits,dim=-1).item()
        print(f"Text: '{text}' --> Predicted Category: {label_map[pred_idx]}")




