import torch

print(f"PyTorch Version: {torch.__version__}")

# Check GPU availability (CUDA for NVIDIA, MPS for Apple Silicon, CPU default)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Create a sample tensor
x = torch.rand(3, 3, device=device)
print("\nSample Tensor (3x3):")
print(x)