import os
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

os.makedirs("models", exist_ok=True)


class LogitsWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def main():
    print("Loading fine-tuned BERT model")
    model = AutoModelForSequenceClassification.from_pretrained("models/bert_finetuned")
    model.eval()
    wrapped = LogitsWrapper(model)
    wrapped.eval()

    dummy_input_ids = torch.ones(1, 16, dtype=torch.long)
    dummy_attention_mask = torch.ones(1, 16, dtype=torch.long)
    onnx_path = "models/bert_classifier.onnx"

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"},
    }

    print(f"Exporting model to {onnx_path}")
    torch.onnx.export(
        wrapped,
        (dummy_input_ids, dummy_attention_mask),
        onnx_path,
        export_params=True,
        opset_version=18,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
    )
    print("BERT ONNX export completed successfully!")


if __name__ == "__main__":
    main()