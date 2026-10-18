import torch
x = torch.tensor(4.0 , requires_grad=True)

y = 3 * x**2 + 2 * x + 1

y.backward()

print(f"Calculated y: {y.item()}")
print(f"Gradient dy/dx at x=4.0: {x.grad.item()}")

assert x.grad.item() == 26.0, "Gradient calculation mismatch!"
print("✅ Autograd working correctly!")