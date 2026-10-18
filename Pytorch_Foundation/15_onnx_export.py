import torch
import torch.nn as nn



class SimpleClassifier(nn.Module):
    def __init__(self , input_dim=10 , num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim,32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32,num_classes)

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model  = SimpleClassifier()
model.eval()

dummy_input = torch.randn(1,10)

onnx_file_path = "models/simple_classifier.onnx"

torch.onnx.export(model,dummy_input,onnx_file_path,export_params=True,opset_version=18,do_constant_folding=True,input_names=["input"] , output_names=["output"],
dynamic_axes={              # Allow flexible batch sizes at runtime
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
print(f"✅ Model exported successfully to ONNX: '{onnx_file_path}'")
