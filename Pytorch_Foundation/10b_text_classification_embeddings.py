import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device {device}")

corpus = [

    ("i loved this movie it was awesome", 2),
    ("fantastic acting and brilliant plot", 2),
    ("highly recommend this masterclass film", 2),

    ("the movie was average and okay", 1),
    ("it was a standard film nothing special", 1),
    ("mediocre plot with decent acting performance", 1),

    ("terrible movie waste of time and money", 0),
    ("horrible script bad acting awful experience", 0),
    ("boring predictable and complete disappointment", 0)
]

vocab = {"<PAD>":0, "<UNK>":1}
for sentence , _ in corpus:
    for word in sentence.split():
        if word not in vocab:
            vocab[word] = len(vocab)

vocab_size = len(vocab)
max_seq_length = 8

def encode_sentence(sentence,vocab,max_len):
    tokens = [vocab.get(word , vocab["<UNK>"]) for word in sentence.split()]
    if len(tokens) < max_len:
        tokens += [vocab["<PAD>"]] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return tokens


class MultiClassTextDataset(Dataset):
    def __init__(self, corpus, vocab, max_len):
        self.data = []
        self.labels = []
        for text, label in corpus:
            self.data.append(encode_sentence(text, vocab, max_len))
            self.labels.append(label)

        self.data = torch.tensor(self.data, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.data[idx] , self.labels[idx]

dataset = MultiClassTextDataset(corpus,vocab,max_seq_length)
train_loader = DataLoader(dataset,batch_size=3,shuffle=True)

class MultiClassLSTM(nn.Module):
    def __init__(self,vocab_size,embed_dim,hidden_dim,output_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size,embed_dim,padding_idx=0)
        self.lstm = nn.LSTM(embed_dim,hidden_dim,batch_first=True,bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2,output_dim)
    def forward(self,x):
        embedded = self.embeddings(x)
        _ , (hidden, _) = self.lstm(embedded)
        hidden_concat = torch.cat((hidden[0], hidden[1]),dim=1)
        out = self.fc(hidden_concat)
        return out

embed_dim = 16
hidden_dim = 32
output_dim =3

model = MultiClassLSTM(vocab_size,embed_dim,hidden_dim,output_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)

epochs = 40
print(f"Training 3-Class text classifier {vocab_size}")

for epoch in range(epochs):
    model.train()
    running_loss = 0
    for x_batch , y_batch in train_loader:
        x_batch , y_batch = x_batch.to(device) , y_batch.to(device)
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}] | Loss: {running_loss / len(train_loader):.4f}")

model.eval()
test_sentences = [
    "brilliant acting and awesome plot",
    "it was an average film and okay",
    "awful waste of money and terrible script"
]

class_map = {0: "Negative 😞", 1: "Neutral 😐", 2: "Positive 😊"}

print("\n--- 3-Class Model Predictions ---")
with torch.no_grad():
    for text in test_sentences:
        encoded = torch.tensor([encode_sentence(text,vocab,max_seq_length)]).to(device)
        output = model(encoded)
        print("Raw output shape",output.shape)
        print("Raw Scores (Logits)", output)
        prediction = torch.argmax(output, dim=1).item()
        print(f"Sentence: '{text}' --> Prediction: {class_map[prediction]}")





