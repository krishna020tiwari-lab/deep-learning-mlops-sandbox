
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:",device)

sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 64
num_epochs = 3
learning_rate = 0.001

transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data',train=False, transform=transforms, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,shuffle=True,batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,shuffle=True,batch_size=batch_size)

class SequenceRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size , hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size,num_classes)

    def forward(self,x):
        x = x.squeeze(1)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out , _ = self.lstm(x,(h0,c0))

        out = self.fc(out[:, -1, :])
        return out
model = SequenceRNN(input_size, hidden_size, num_layers, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("Training Loop")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images , labels in train_loader:
        images , labels = images.to(device) , labels.to(device)

        outputs = model(images)
        loss = criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epochs [{epoch + 1} / {num_epochs}] | Loss: {running_loss / len(train_loader):.4f}")

model.eval()
total , correct = 0,0
with torch.no_grad():
    for images , labels in test_loader:
        images , labels = images.to(device) , labels.to(device)
        outputs = model(images)
        _ , predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"\nAccuracy on Test Set: {100 * correct / total:.2f}%")



