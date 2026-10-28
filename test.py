import numpy as np
import io
from pathlib import Path
from fastapi import FastAPI,HTTPException,File,UploadFile
from PIL import Image
import onnxruntime as ort
from pydantic import BaseModel
from transformers import AutoTokenizer
from contextlib import asynccontextmanager

BASE_DIR  = Path(__file__).resolve().parent # base_dir is the folder where file lives

# This is for the path for two models
VISION_MODEL_PATH = BASE_DIR / "models" / "resnet18_single.onnx"
TEXT_MODEL_PATH = BASE_DIR / "models" / "bert_classifier.onnx"
TOKENIZER_PATH = BASE_DIR / "models" / "bert_tokenizer"

# this is the empty placeholders they are here for so model runs at once instead of request and None means not loaded yet
vision_session = None
text_session = None
tokenizer = None

# The average and spread of the RGB channels in the ImageNet dataset resnet18 normalized on these number so we also have to normalize at inference float32 is the type of onnx

IMAGENET_MEAN = np.array([0.485,0.456,0.406] , dtype=np.float32)
IMAGENET_STD = np.array([0.229,0.224,0.225] , dtype=np.float32)

# It is the index to name like negative 0 neutral 1 positive 2
TEXT_LABELS = ["Negative" , "Neutral" , "Positive"]

# small dictionary of ImageNet class and numbers ImagenNet has 1000 classes this includes only five so the rest can fall back to ImageNet Class
SAMPLE_LABELS = {
    0: "tench",
    207: "Golden Retriever",
    263: "Pembroke Welsh Corgi",
    281: "tabby cat",
    285: "Egyptian cat",
}

# Define the function runs once when the server start (the code before yield) and once when it stops (the code after yield)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Without global we have to create a mew local variable withIt we modify-level ones
    global vision_session,text_session,tokenizer
    # this is for to check that file is on disk or not , which avoids crash if it is missing
    if VISION_MODEL_PATH.exists():
        # str is use because onnx wants a string path CPUExecutionProvider means it runs on CPU
        vision_session = ort.InferenceSession(str(VISION_MODEL_PATH),providers=["CPUExecutionProvider"])
        print("Vision onnx model is loaded",flush=True)
    else:
        print(f"Warning vision model not found at {VISION_MODEL_PATH}",flush=True)

    if TEXT_MODEL_PATH.exists():
        text_session = ort.InferenceSession(str(TEXT_MODEL_PATH),providers=["CPUExecutionProvider"])
        print("Text onnx model loaded",flush=True)
    else:
        print(f"Warning text model not found at {TEXT_MODEL_PATH}",flush=True)
    if TOKENIZER_PATH.exists():
        # it loads my saved tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH))
    else:
        # it downloads the bert-base-uncased from the internet
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    print("Loaded Tokenizer",flush=True)
    yield # yield is the dividing line the server runs server requests while used here
    # It runs when the server stop
    print("Shutting Down Unified AI Service",flush=True)

# Create the app title and version in auto docs lifespan is used for startup and shutdown function
app = FastAPI(title="Unified Vision & Text ONNX Service",version="1.0.0",lifespan=lifespan)

# It defines what the image endpoint return. FastAPI validates the response against it and documents it
# like this {"status":"success","filename":"dog.jpg","predicted_class":"207","label_description":"Golden Retriever","confidence":0.93}
class ImagePredictionResponse(BaseModel):
    status:str
    filename:str
    predicted_class:str
    label_description:str
    confidence:float

class TextPayload(BaseModel):
    text:str

# define the input text endpoint {"status":"success","input_text":"I love this","predicted_class":2,"label":"Positive","probabilities":[0.01,0.04,0.95]}.
class TextPredictionResponse(BaseModel):
    status:str
    input_text:str
    predicted_class:int
    label:str
    probabilities:list[float]

# This turns raw model(logits) into probabilities that sum to 1 Turns raw model scores (logits) into probabilities that sum to 1
# . Example: [2.0, 1.0, 0.1] becomes about [0.66, 0.24, 0.10].
def softmax(x:np.ndarray) -> np.ndarray:
    # computing e^x subtracting the max first prevents overflow with huge numbers and does not change the result axis=-1 means along the last dimension (the classes)
    e_x = np.exp(x - np.max(x,axis=-1,keepdims=True))
    # Divides each value to the total , so everything sums to up
    return e_x / np.sum(e_x , axis=-1,keepdims=True)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    # This converts raw image bytes into exact tensor the model expects
    # io.BytesIO(image_bytes) wraps the bytes as a file.
    # Image.open(...) decodes the image.
    # .convert("RGB") forces 3 channels, which handles PNGs with transparency (RGBA) or grayscale.
    # .resize((224, 224)) matches the model’s input size.
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224,224))
    # converts the image to an array of shape (224,224,3) with values 0 to 255 the scales to 0.0 to 1.0
    img_array = np.array(image,dtype=np.float32) / 255.0
    # Normalization subtract the mean and divide by std per channel numpy broadcasts 3-values array across the last dimension
    img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD
    # reorders axes from HWC (Height,Weight,Channel) to CHW,because Pytorch-trained models expect channel first So the shape becomes this (3,224,224)
    img_array = np.transpose(img_array,(2,0,1))
    # this is use for to add a batch at the front of the dimension models expect a batch even for a single image we also make sure that the type is right
    return np.expand_dims(img_array,axis=0).astype(np.float32)

# it registers a get route open the url and see server is alive
@app.get("/")
def root():
    return {"service":"Unified ONNX Inference API","status":"online"}

# MLOps HealthCheckpoint route: verifies the model session  and server are loaded
@app.get("/health")
def health_check():
    # This is for check that models is initialized or not
    if text_session is None or vision_session is None:
        raise HTTPException(status_code=503,detail="Models are not Initialized yet")
    return {"status":"healthy","vision_model_loaded":vision_session is not None,"text_session":text_session is not None}
# @app.post(...) registers a POST route.
# response_model enforces the output shape.
# async def is used because we await the file read.
# File(...) means the upload is required.
@app.post("/predict/image" , response_model=ImagePredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    if vision_session is None:
        raise HTTPException(status_code=503,detail="Vision Model is not Loaded") # 503 is means service unavailable since model is nnot loaded
    if file.content_type not in ["image/jpeg" , "image/jpg", "image/png"]:
        raise HTTPException(status_code=400,detail="Please a jpg or png file") # Checking the file type 400 mean bad request
    try:
        # read the whole upload file into bytes and await frees the server to handle other requests while waiting
        file_bytes = await file.read()
        # Runs the Preprocessing
        input_tensor = preprocess_image(file_bytes)
        # It asks the ONNX Model for it's input name (for example "input")
        input_name = vision_session.get_inputs()[0].name
        # First argument None means “return all outputs”.
        # Second argument is a dict of {input name: array}.
        # run returns a list of outputs, so [0] takes the first output, shape (1, 1000).
        # The second [0] takes the first image in the batch, shape (1000,).
        logits = vision_session.run(None,{input_name: input_tensor})[0][0]
        # Convert to probabilities the argmax gives the index of the highest one int() converts from numpy into normal python int
        probs = softmax(logits)
        pred_class = int(np.argmax(probs))
        # dict.get(key, default) returns the name if known, otherwise the fallback text.
        label_desc = SAMPLE_LABELS.get(pred_class,f'"ImageNet Class #{pred_class}')
        # Build The response str(pred_class) is used because the schema says str.float(...) converts from Numpy float,and round(...,4) keeps 4 decimal
        return ImagePredictionResponse(
            status="success",
            filename=file.filename,
            predicted_class=str(pred_class),
            label_description=label_desc,
            confidence=round(float(probs[pred_class]),4),)
    except Exception as e:
        raise HTTPException(status_code=500,detail=f"Image Processing Error: {str(e)}")

@app.post("/predict/text",response_model=TextPredictionResponse)
# See this is a normal def unlike async def because there is no awaiting FastAPI runs it in thread pool payload is parsed from the JSON body into TextPayload
def predict_text(payload: TextPayload):
    # Checks that model is loaded or not and rejects empty or whitespace
    if text_session is None:
        raise HTTPException(status_code=503,detail="Text Model is not loaded")
    if not payload.text.strip():
        raise HTTPException(status_code=400,detail="Input Text cannot be empty")
# Tokenizes the text. Example: "I love it" becomes [101, 1045, 2293, 2009, 102, 0, 0, ...].
    #
    # max_length=64 sets the fixed length.
    # padding="max_length" pads short text with 0s up to 64.
    # truncation=True cuts longer text down to 64.
    # return_tensors="np" returns NumPy arrays (not PyTorch).
    try:
        inputs = tokenizer(
        payload.text,
        max_length=64,
        padding="max_length",
        truncation=True,
        return_tensors="np",

    )
        # It's build the ONNX dict
        # input_ids are the token numbers
        # attention_mask is 1 for real tokens and 0 for bert so bert ignore padding
        # int64 is what ONNX expects
        ort_inputs = {"input_ids":inputs["input_ids"].astype(np.int64),"attention_mask":inputs["attention_mask"].astype(np.int64)}
        logits = text_session.run(None,ort_inputs)[0][0]
        probs = softmax(logits).tolist()
        pred_class = int(np.argmax(probs))
        return TextPredictionResponse(
            status="success",
            input_text=payload.text,
            predicted_class=pred_class,
            label=TEXT_LABELS[pred_class] if pred_class < len(TEXT_LABELS) else f"Class {pred_class}",
            probabilities=[round(p,4) for p in probs],)
    except Exception as e:
        raise HTTPException(status_code=500,detail=f"Text Processing Error {str(e)}")


























