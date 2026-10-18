import os
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
import onnxruntime as ort
from transformers import AutoTokenizer


base_dir = Path(__file__).resolve().parent.parent
model_path = base_dir / "models" / "bert_classifier.onnx"
tokenizer_path = 'models/bert_tokenizer'
labels = ["Negative (Class 0)", "Positive (Class 1)"]

ort_session = None
tokenizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ort_session , tokenizer

    if os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX file was not found {model_path}")

    ort_session = ort.InferenceSession(model_path , providers=["CPUExecutionProvider"])
    print("🚀 ONNX Runtime Inference Engine & Tokenizer Loaded Successfully!")

    yield
app = FastAPI(title="BERT text Classification ONNX Microservice",description="Production REST API serving Hugging Face BERT exported to ONNX Runtime",
    version="1.0.0",lifespan=lifespan)

class TextPayload(BaseModel):
    text:str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"text": "This deep learning and MLOps course is absolutely fantastic!"}
            ]
        }
    }


class PredictionResponse(BaseModel):
    status:str
    input_text:str
    prediction_class:str
    label:str
    probabilities:list[float]

@app.get("/")
def root():
    return {
        "service":"BERT ONNX Inference API",
        "status": "Online",
        "docs_url": "/docs"
    }
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1,keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

@app.post("/predict" , response_model=PredictionResponse)
def predict(payload: TextPayload):
    if not payload.text.strip():
        raise HTTPException(status_code=400,detail="Input text cannot be empty")

    try:
        inputs = tokenizer(payload.text,max_length=64,padding="max_length",truncation=True,return_tensors="np")

        input_ids = inputs["input_ids"].astype(np.int64)
        attention_mask = inputs["attention_mask"].astype(np.int64)
        ort_inputs = {
            "inputs_ids": input_ids,
            "attention_mask": attention_mask
        }
        logits = ort_session.run(None , ort_inputs)[0][0]
        probs = softmax(logits).tolist()
        predicted_class = int(np.argmax(probs))

        return PredictionResponse(
            status="Success",
            input_text=payload.text,
            predicted_class = predicted_class,
            labels = labels[predicted_class] if predicted_class < len(labels) else f"Class {predicted_class}",
            probabilities=[round(p,4) for p in probs]
        )

    except Exception as e:
        raise HTTPException(status_code=500,detail=f"Inference error: {str(e)}")


