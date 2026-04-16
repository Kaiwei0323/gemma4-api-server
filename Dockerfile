# CUDA + PyTorch runtime (GPU). Host needs NVIDIA Container Toolkit / Docker GPU support.
# Model weights are NOT baked in: mount your checkpoint at /model (see docker-compose.yml).
#
# Force all weights on GPU 0 (OOM if checkpoint does not fit in VRAM at current dtype):
#   GEMMA_DEVICE_MAP=cuda0 (set below)
# If OOM: use GEMMA_DEVICE_MAP=auto or GEMMA_LOAD_4BIT=1 (see api_server.py).

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GEMMA_REQUIRE_GPU=1 \
    GEMMA_DEVICE_MAP=cuda0 \
    GEMMA_MODEL_PATH=/model \
    GEMMA_IMAGES_DIR=/images \
    GEMMA_WEIGHTS_TQDM=1 \
    GEMMA_LOAD_HEARTBEAT=1 \
    GEMMA_LOAD_HEARTBEAT_SEC=10

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .
RUN pip install --upgrade pip && pip install -r requirements-docker.txt

COPY api_server.py .

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=420s --retries=2 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=5).read()"

CMD ["python", "-m", "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "5000", "--log-level", "info"]
