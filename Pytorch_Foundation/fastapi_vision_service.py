import io
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
from transformers import AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent

VISION_MODEL_PATH = BASE_DIR / "models" / "resnet18_single.onnx"
TEXT_MODEL_PATH = BASE_DIR / "models" / "bert_classifier.onnx"      # check real filename
TOKENIZER_PATH = BASE_DIR / "models" / "bert_tokenizer"

vision_session = None
text_session = None
tokenizer = None

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TEXT_LABELS = ["Negative", "Neutral", "Positive"]
SAMPLE_LABELS = {
    0: "tench",
    207: "Golden Retriever",
    263: "Pembroke Welsh Corgi",
    281: "tabby cat",
    285: "Egyptian cat",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vision_session, text_session, tokenizer

    if VISION_MODEL_PATH.exists():
        vision_session = ort.InferenceSession(str(VISION_MODEL_PATH), providers=["CPUExecutionProvider"])
        print("Vision ONNX Model Loaded", flush=True)
    else:
        print(f"WARNING: vision model not found at {VISION_MODEL_PATH}", flush=True)

    if TEXT_MODEL_PATH.exists():
        text_session = ort.InferenceSession(str(TEXT_MODEL_PATH), providers=["CPUExecutionProvider"])
        print("Text ONNX Model Loaded", flush=True)
    else:
        print(f"WARNING: text model not found at {TEXT_MODEL_PATH}", flush=True)

    if TOKENIZER_PATH.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH))
    else:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    print("Loaded Tokenizer", flush=True)

    yield
    print("Shutting down Unified AI Service", flush=True)


app = FastAPI(title="Unified Vision & Text ONNX Service", version="1.0.0", lifespan=lifespan)


class ImagePredictionResponse(BaseModel):
    status: str
    filename: str
    predicted_class: str
    label_description: str
    confidence: float


class TextPayload(BaseModel):
    text: str


class TextPredictionResponse(BaseModel):
    status: str
    input_text: str
    predicted_class: int
    label: str
    probabilities: list[float]


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD
    img_array = np.transpose(img_array, (2, 0, 1))
    return np.expand_dims(img_array, axis=0).astype(np.float32)


@app.get("/")
def root():
    return {"service": "Unified ONNX Inference API", "status": "online"}

@app.get("/health")
def health():
    if text_session is None or vision_session is None:
        raise HTTPException(status_code=503,detail="Models is not Initialized")
    return {"status":"healthy","vision_model_loaded":vision_session is not None,"text_session_loaded":text_session is not None}




@app.post("/predict/image", response_model=ImagePredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    if vision_session is None:
        raise HTTPException(status_code=503, detail="Vision model is not loaded")
    if file.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
        raise HTTPException(status_code=400, detail="Please upload a JPEG or PNG image")

    try:
        file_bytes = await file.read()
        input_tensor = preprocess_image(file_bytes)
        input_name = vision_session.get_inputs()[0].name
        logits = vision_session.run(None, {input_name: input_tensor})[0][0]

        probs = softmax(logits)
        pred_class = int(np.argmax(probs))
        label_desc = SAMPLE_LABELS.get(pred_class, f"ImageNet Class #{pred_class}")

        return ImagePredictionResponse(
            status="success",
            filename=file.filename,
            predicted_class=str(pred_class),
            label_description=label_desc,
            confidence=round(float(probs[pred_class]), 4),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image Processing Error: {str(e)}")


@app.post("/predict/text", response_model=TextPredictionResponse)
def predict_text(payload: TextPayload):
    if text_session is None:
        raise HTTPException(status_code=503, detail="Text model is not loaded")
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    try:
        inputs = tokenizer(
            payload.text,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        logits = text_session.run(None, ort_inputs)[0][0]
        probs = softmax(logits).tolist()
        pred_class = int(np.argmax(probs))

        return TextPredictionResponse(
            status="success",
            input_text=payload.text,
            predicted_class=pred_class,
            label=TEXT_LABELS[pred_class] if pred_class < len(TEXT_LABELS) else f"Class {pred_class}",
            probabilities=[round(p, 4) for p in probs],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text Processing Error: {str(e)}")