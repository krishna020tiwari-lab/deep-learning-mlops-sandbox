┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Define Schemas (Pydantic Request & Response Models)             │
├─────────────────────────────────────────────────────────────────────────┤
│ STEP 2: Lifespan Startup (Load ONNX Model & Preprocessors into RAM)     │
├─────────────────────────────────────────────────────────────────────────┤
│ STEP 3: API Endpoint (Receive Raw Request Payload)                      │
├─────────────────────────────────────────────────────────────────────────┤
│ STEP 4: Preprocess (Transform Raw Data ──► NumPy Tensor Arrays)         │
├─────────────────────────────────────────────────────────────────────────┤
│ STEP 5: ONNX Execution (Execute ort_session.run)                      │
├─────────────────────────────────────────────────────────────────────────┤
│ STEP 6: Postprocess (Transform Output Logits ──► Probabilities/Labels) │
├─────────────────────────────────────────────────────────────────────────┤
│ STEP 7: Return Response (Serialize JSON Response Output)                │
└─────────────────────────────────────────────────────────────────────────┘