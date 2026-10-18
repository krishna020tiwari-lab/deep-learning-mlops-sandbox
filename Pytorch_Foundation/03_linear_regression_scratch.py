import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
X = torch.randn(100, 1, device=device)
y_true = 3 * X + 2 + torch.randn(100, 1, device=device) * 0.1

w = torch.randn(1,1, requires_grad=True,device=device)
b = torch.randn(1,1, requires_grad=True,device=device)

learning_rate = 0.1
epochs = 100

print(f"Initial w: {w.item():.4f}, Initial b: {b.item():.4f}\n")

for epoch in range(epochs):
    y_pred = X @ w + b

    loss = torch.mean((y_pred - y_true))
    loss.backward()

    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

        w.grad.zero_()
        b.grad.zero_()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}] | Loss: {loss.item():.4f} | w: {w.item():.4f} | b: {b.item():.4f}")

print("\n--- Training Complete ---")
print(f"True Equation:  y = 3.0000 * x + 2.0000")
print(f"Learned Params: y = {w.item():.4f} * x + {b.item():.4f}")
