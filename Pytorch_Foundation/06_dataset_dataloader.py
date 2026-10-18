import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class QuadraticDataset(Dataset):
    def __init__(self, num_samples=1000):
        torch.manual_seed(42)
        self.x = torch.unsqueeze(torch.linspace(-3,3,num_samples), dim=1)
        self.y = self.x.pow(2) + torch.randn(self.x.size()) * 0.2

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


dataset = QuadraticDataset(num_samples=1000)
train_loader = DataLoader(dataset=dataset,batch_size=32,shuffle=True)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1,16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Linear(16,1),
        )

    def forward(self, x):
        return self.net(x)

model = MLP().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()


epochs = 20
print(f"Training on {len(dataset)} samples across {len(train_loader)} batches per epoch...\n")

for epoch in range(epochs):
    epoch_loss = 0
    for batch_x ,batch_y in train_loader:
        batch_x,batch_y = batch_x.to(device), batch_y.to(device)
        prediction = model(batch_x)
        loss = criterion(prediction, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_x.size(0)

    avg_loss = epoch_loss / len(dataset)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}] | Avg Loss: {avg_loss:.4f}")

print("\n✅ Mini-batch training completed successfully!")




