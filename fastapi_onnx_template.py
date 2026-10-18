import os
from contextlib import asynccontextmanager
from pathlib import Path
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# =====================================================================
# PATH CONFIGURATION
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "your_model.onnx"
LABELS = ["Class_0", "Class_1", "Class_2"]

# Global Handles
ort_session = None


# =====================================================================
# STEP 1: DEFINE PYDANTIC SCHEMAS
# =====================================================================
class InferenceRequest(BaseModel):
    # Adjust fields based on input type (e.g., text: str, features: list[float], etc.)
    features: list[float]

    model_config = {
        "json_schema_extra": {
            "examples": [{"features": [0.5, -1.2, 0.3, 0.8]}]
        }
    }


class InferenceResponse(BaseModel):
    status: str
    predicted_class: int
    label: str
    probabilities: list[float]


# =====================================================================
# STEP 2: LIFESPAN MODEL LOADER (Server Startup/Shutdown)
# =====================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ort_session
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"ONNX Model missing at: {MODEL_PATH}")

    # Load ONNX Inference Session once at startup
    ort_session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    print(f"🚀 Loaded ONNX Model: {MODEL_PATH}")

    yield
    print("🛑 Service Shutdown Complete.")


app = FastAPI(
    title="Universal ONNX Microservice",
    version="1.0.0",
    lifespan=lifespan
)


# =====================================================================
# MATH HELPER FUNCTIONS
# =====================================================================
def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


# =====================================================================
# STEPS 3–7: PREDICTION ROUTE
# =====================================================================
@app.post("/predict", response_model=InferenceResponse)
def predict(payload: InferenceRequest):
    try:
        # -------------------------------------------------------------
        # STEP 4: PREPROCESS (Raw Input ──► NumPy Array)
        # -------------------------------------------------------------
        input_tensor = np.array([payload.features], dtype=np.float32)

        # -------------------------------------------------------------
        # STEP 5: ONNX EXECUTION
        # -------------------------------------------------------------
        input_name = ort_session.get_inputs()[0].name
        logits = ort_session.run(None, {input_name: input_tensor})[0][0]

        # -------------------------------------------------------------
        # STEP 6: POSTPROCESS (Logits ──► Probabilities & Class)
        # -------------------------------------------------------------
        probs = softmax(logits).tolist()
        predicted_class = int(np.argmax(probs))

        # -------------------------------------------------------------
        # STEP 7: RETURN JSON RESPONSE
        # -------------------------------------------------------------
        return InferenceResponse(
            status="success",
            predicted_class=predicted_class,
            label=LABELS[predicted_class] if predicted_class < len(LABELS) else f"Class_{predicted_class}",
            probabilities=[round(p, 4) for p in probs]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Failure: {str(e)}")