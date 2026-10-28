from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import importlib

service_module = importlib.import_module("Pytorch_Foundation.fastapi_vision_service")
app = service_module.app


def test_health_endpoint():
    fake_session = MagicMock()
    fake_tokenizer = MagicMock()

    with patch("os.path.exists", return_value=True), \
         patch("os.path.isfile", return_value=True), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.is_file", return_value=True), \
         patch("onnxruntime.InferenceSession", return_value=fake_session), \
         patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tokenizer):
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json() == {
                "status": "healthy",
                "vision_model_loaded": True,
                "text_session_loaded": True,
            }