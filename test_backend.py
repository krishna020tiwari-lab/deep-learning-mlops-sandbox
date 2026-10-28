from fastapi.testclient import TestClient
from Pytorch_Foundation.fastapi_vision_service import app
import importlib
module = importlib.import_module("Pytorch_Foundation.fastapi_vision_service")
app = module.app

client = TestClient(app)

def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy","vision_model_loaded":True,"text_session_loaded":True}