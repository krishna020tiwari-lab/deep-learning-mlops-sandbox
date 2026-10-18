import numpy as np
from pathlib import Path
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="ONNX Classifier API" , description="A lightweight API serving an exported ONNX model using ONNX Runtime.",
    version="1.0")

BASE_DIR = Path(__file__).resolve().parent
onnx_model_path = BASE_DIR / "models" / "simple_classifier.onnx"

try:
    session = ort.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"onnx initialized successfully from {onnx_model_path}")
except Exception as e:
    print(f"❌ Failed to load ONNX model: {e}")
    session = None

class PredictRequest(BaseModel):
    features: list[float]

@app.get("/")
def read_root():
    return {"message": "ONNX Model Serving API is live! Navigate to /docs for interactive API testing."}

@app.post("/predict")
def predict(request: PredictRequest):
    if session is None:
        raise HTTPException(status_code=500,detail="ONNX Model is not loaded")

    if len(request.features) != 10:
        raise HTTPException(status_code=400,detail=f"Expected 10 input features, but received {len(request.features)}")

    input_data = np.array([request.features] , dtype=np.float32)
    raw_outputs = session.run([output_name] , {input_name: input_data})
    logits = raw_outputs[0][0].tolist()

    predicted_class = int(np.argmax(logits))

    return {
        "status": "success",
        "predicted_class": predicted_class,
        "logits": logits
    }


