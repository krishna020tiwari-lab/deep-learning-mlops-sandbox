from unittest.mock import patch
from fastapi.testclient import TestClient
import importlib
service_module = importlib.import_module("Pytorch_Foundation.fastapi_vision_service")
app = service_module.app

client = TestClient(app)

def test_health_endpoint():
    with patch("os.path.exists" , return_value=True), \
        patch("onnxruntime.InferenceSession",return_value=True):
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json() == {"status": "healthy","vision_model_loaded":True,"text_session_loaded":True}

            data = response.json()
            assert data["status"] == "healthy"