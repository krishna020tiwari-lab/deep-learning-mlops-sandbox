import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Setup CUDA(Compute Unified Data Architecture) Device

# Generate Synthetic Data
torch.manual_seed(42)
X = torch.randn(100,1,device=device)
Y_true = 3 * X + 2 + torch.randn(100,1,device=device) * 0.1

# Define the model class using nn.Module
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Linear automatically creates weights (w) and bias (b) for y = Xx + b
        self.Linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        return self.Linear(x)

# Instantiate Model, Loss Function , Optimizer
model = LinearRegressionModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01) # Stochastic Gradient Descent

print("Initial Parameters:")
for name , param in model.named_parameters():
    print(f"{name}: {param.data.squeeze().item():.4f}")
print("\nTraining")

# Clean Training Loop
epochs = 100
for epoch in range(epochs):
    # Forward pass
    y_pred = model(X)
    loss = criterion(y_pred, Y_true)

    # Backward pass & Optimization
    optimizer.zero_grad() #Reset old gradients
    loss.backward() # Autograd backpropagation
    optimizer.step() # Update parameters automatically

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}] | Loss: {loss.item():.4f}")

print("\n---Model Trained Successfully---")
# Extract trained weights and bias
w_learned = model.Linear.weight.item()
b_learned = model.Linear.bias.item()

print(f"True Equation: y = 3.0000 * x + 2.0000")
print(f"Learned Params: {w_learned:.4f} * x + {b_learned:.4f}")


