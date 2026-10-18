import torch
import torch.nn as nn
import torch.optim as optim

device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(42)
X = torch.unsqueeze(torch.linspace(-3,3,200 , device=device) ,dim=1)
y_true = X.pow(2) + torch.randn(X.size(),device=device)

class NonLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=1, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=1)
        )
    def forward(self, x):
        return self.net(x)

model = NonLinearModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


epochs = 500
for epoch in range(epochs):
    y_pred = model(X)
    loss = criterion(y_pred, y_true)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1 ) % 5 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}] | Loss: {loss.item():.4f}")

print("\n✅ MLP Model Successfully Trained on Non-Linear Quadratic Data!")

