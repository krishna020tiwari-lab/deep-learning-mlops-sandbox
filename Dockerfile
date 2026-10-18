FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn onnxruntime pillow numpy python-multipart transformers

COPY Pytorch_Foundation/20_fastapi_vision_service.py /app/Pytorch_Foundation/20_fastapi_vision_service.py
COPY Pytorch_Foundation/models /app/Pytorch_Foundation/models

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "--app-dir", "Pytorch_Foundation", "20_fastapi_vision_service:app", "--host", "0.0.0.0", "--port", "8000"]