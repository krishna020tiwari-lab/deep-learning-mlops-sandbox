import os
import torch
import torchvision.models as models

os.makedirs("models",exist_ok=True)

def main():
    print("Loading pre-trained ResNet18 model.....")
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval()

    dummy_input = torch.randn(1,3,224,224)
    onnx_path = "models/resnet18.onnx"

    dynamic_axes = {
        "image_input":{0:"batch_size"},
        "logits":{0:"batch_size"}
    }

    print(f"Exporting ResNet18 to {onnx_path}")
    torch.onnx.export(model,dummy_input,onnx_path,export_params=True,opset_version=14,do_constant_folding=True,input_names=["image_input"],output_names=["logits"],
                      dynamic_axes=dynamic_axes)

    print("RESNET18 ONNX Export Completed Successfully")

if __name__ == "__main__":
    main()

