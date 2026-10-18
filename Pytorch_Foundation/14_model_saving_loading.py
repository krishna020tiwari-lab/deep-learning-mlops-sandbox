import os
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device {device}")

class SimpleClassifier(nn.Module):
    def __init__(self , input_dim=10,num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim,32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32,num_classes)

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

torch.manual_seed(42)
original_model = SimpleClassifier().to(device)
original_model.eval()

sample_input = torch.randn(1,10).to(device)

with torch.no_grad():
    original_output = original_model(sample_input)

print("\n--- Baseline Forward Pass ---")
print("Original Model Output Logits:", original_output)

os.makedirs("models",exist_ok=True)
save_path = "models/simple_classifier_weights.pth"

torch.save(original_model.state_dict() , save_path)
print(f"\n✅ Model state_dict successfully saved to: '{save_path}'")

loaded_model = SimpleClassifier().to(device)
state_dict = torch.load(save_path,map_location=device)

loaded_model.load_state_dict(state_dict)
loaded_model.eval()
with torch.no_grad():
    loaded_output = loaded_model(sample_input)

print("\n--- Verification After Reloading ---")
print("Loaded Model Output Logits:  ", loaded_output)

are_equal = torch.allclose(original_output, loaded_output)
print(f"\nOutputs Match Identically? -> {are_equal} 🎉")
