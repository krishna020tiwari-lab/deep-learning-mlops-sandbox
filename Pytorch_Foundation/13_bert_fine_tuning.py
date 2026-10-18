import torch
from torch.utils.data import Dataset , DataLoader
from transformers import AutoTokenizer , AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device {device}")

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_texts = [
    # negative (0)
    "Terrible experience, complete waste of time.",
    "Horrible script and awful acting.",
    "Worst film ever made, complete garbage.",
    "I hated every minute, so boring and bad.",
    "The plot was a mess and the acting was painful.",
    "Disappointing and dull, I regret watching it.",
    "A total disaster, avoid this movie.",
    "Awful dialogue and terrible pacing.",
    # neutral (1)
    "It was okay, nothing special.",
    "An average film, neither bad nor good.",
    "It was an acceptable movie, I guess.",
    "Some parts were fine, some were boring.",
    "Not great, not terrible, just average.",
    "A decent watch but nothing memorable.",
    "The movie was fine, I felt neutral about it.",
    "Mediocre story with ordinary acting.",
    # positive (2)
    "Great movie, loved every second!",
    "Brilliant performance and awesome plot!",
    "This movie is amazing and wonderful.",
    "Fantastic story, I absolutely loved it.",
    "An excellent film with beautiful acting.",
    "What a masterpiece, truly outstanding.",
    "I enjoyed it so much, highly recommended.",
    "Superb direction and a touching story.",
]



train_labels = [0]*8 + [1]*8 + [2]*8

class SentimentDataset(Dataset):
    def __init__(self,texts,labels,tokenizer,max_len=32):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

dataset = SentimentDataset(train_texts,train_labels,tokenizer)
dataloader = DataLoader(dataset,batch_size=2,shuffle=True)

model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=3).to(device)

optimizer = torch.optim.AdamW(model.parameters(),lr=2e-5)

epochs = 12
print("\n--- Starting BERT Fine-Tuning ---")
model.train()

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
test_sentences = [
    "Awesome cinematography and fantastic story",
    "It was an acceptable and average movie",
    "Worst film ever made, complete garbage"
]

id2label = {0: "Negative 😞", 1: "Neutral 😐", 2: "Positive 😊"}

print("\n--- Model Predictions ---")
with torch.no_grad():
    for text in test_sentences:
        inputs= tokenizer(text,return_tensors="pt",truncation=True,padding=True).to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        pred_idx = torch.argmax(logits , dim=1).item()

        print(f"Sentence: '{text}' --> Prediction: {id2label[pred_idx]}")
model.save_pretrained("models/bert_finetuned")
tokenizer.save_pretrained("models/bert_tokenizer")
print("Saved fine-tuned model and tokenizer")