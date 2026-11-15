import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device {device}")

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

sentences = [
    "This MLOps pipeline setup was smoothly executed!",
    "The deployment process was confusing and frustrating."
]

inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(device)

model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.softmax(outputs.logits, dim=1)

for text , prob in zip(sentences,prediction):
    label_idx = torch.argmax(prob).item()
    label = model.config.id2label[label_idx]
    confidence = prob[label_idx].item()
    print(f"Text: '{text}' -> Sentiment: {label} ({confidence:.2%})")