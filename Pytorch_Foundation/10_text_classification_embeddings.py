import torch
import torch.nn as nn
import torch.optim as optim
from networkx.algorithms import similarity
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset , DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device {device}")

corpus = [
    ("i loved this movie it was awesome", 1),
    ("fantastic acting and great plot", 1),
    ("highly recommend this masterclass film", 1),
    ("truly brilliant and inspiring storyline", 1),
    ("terrible movie waste of time and money", 0),
    ("horrible script bad acting awful experience", 0),
    ("boring predictable and complete disappointment", 0),
    ("worst film ever created useless attempt", 0)
]

vocab = {"<PAD>": 0, "<UNK>" : 1}
for sentence , _ in corpus:
    for word in sentence.split():
        if word not in vocab:
            vocab[word] = len(vocab)

vocab_size = len(vocab)
max_seq_length = 8

def encode_sentence(sentence,vocab , max_len):
    tokens = [vocab.get(word, vocab["<UNK>"]) for word in sentence.split()]
    if len(tokens) < max_len:
        tokens +=[vocab["<PAD>"]] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return tokens

class TextDataset(Dataset):
    def __init__(self,corpus,vocab,max_len):
        self.data = []
        self.labels = []
        for text , label in corpus:
            self.data.append(encode_sentence(text,vocab,max_len))
            self.labels.append(label)

        self.data = torch.tensor(self.data,dtype=torch.long)
        self.labels = torch.tensor(self.labels,dtype=torch.long)

    def __len__(self):
        return len(self.labels)
    def __getitem__(self,idx):
        return self.data[idx] , self.labels[idx]

dataset = TextDataset(corpus,vocab,max_seq_length)
train_loader = DataLoader(dataset,batch_size=2,shuffle=True)

class SentimentLSTM(nn.Module):
    def __init__(self,vocab_size,embed_dim,hidden_dim,output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embed_dim,padding_idx=0)
        self.lstm = nn.LSTM(embed_dim,hidden_dim,batch_first=True)
        self.fc = nn.Linear(hidden_dim,output_dim)

    def forward(self,x):
        embedded = self.embedding(x)
        _ , (hidden,_) = self.lstm(embedded)
        out = self.fc(hidden.squeeze(0))
        return out

embed_dim = 16
hidden_dim = 32
output_dim = 2

model = SentimentLSTM(vocab_size,embed_dim,hidden_dim,output_dim).to(device)
criterion = CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)

epochs = 30
print(f"\nTraining Text Classification LSTM (Vocab Size: {vocab_size})...\n")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x_batch , y_batch in train_loader:
        x_batch , y_batch = x_batch.to(device) , y_batch.to(device)


        outputs = model(x_batch)
        loss = criterion(outputs,y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}] | Loss: {total_loss / len(train_loader):.4f}")


model.eval()
test_sentences = [
    "awesome script and brilliant movie",
    "terrible waste and awful experience"
]
print("\n--- Model Predictions ---")
with torch.no_grad():
    for text in test_sentences:
        encoded = torch.tensor([encode_sentence(text,vocab,max_seq_length)]).to(device)
        output = model(encoded)
        prediction = torch.argmax(output,dim=1).item()
        label_str = "Positive 😊" if prediction == 1 else "Negative 😞"
        print(f"Sentence: '{text}' --> Prediction: {label_str}")


print("\n--- Challenge 1: Inspecting Learned Word Vectors")

embeddings_weights = model.embedding.weight.data

print(f"Embedding Weight Matrix Shape: {embeddings_weights.shape}")

idx_brilliant = vocab.get("brilliant")
idx_awesome = vocab.get("awesome")

vector_brilliant = embeddings_weights[idx_brilliant]
vector_awesome = embeddings_weights[idx_awesome]

print(f"\nWord : 'brilliant' (Index {idx_brilliant})")
print(f"Vector (size {len(vector_brilliant)}),:\n{vector_brilliant}")

print(f"\nWord : 'awesome' (Index {idx_awesome})")
print(f"Vector (size {len(vector_awesome)}),:\n{vector_awesome}")

similarity = torch.cosine_similarity(vector_brilliant.unsqueeze(0),vector_awesome.unsqueeze(0)).item()

print(f"Cosine Similarity Between 'brilliant' and 'awesome': {similarity:.4f}")