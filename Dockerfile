# Gemma 4 API with vLLM (AsyncLLM) for /chat/stream, /image/stream, /video/stream.
# Base image provides CUDA 13.0 + vLLM tuned for Gemma 4.
# Mount your full checkpoint at /model (see docker-compose.yml).

FROM vllm/vllm-openai:gemma4-cu130

# Base image sets ENTRYPOINT to the `vllm` CLI; clear it so CMD runs Python/uvicorn.
ENTRYPOINT []

# Base image may not provide a `python` name on PATH (conda often has /opt/conda/bin/python only).
RUN set -eux; \
    if command -v python >/dev/null 2>&1; then exit 0; fi; \
    if [ -x /opt/conda/bin/python ]; then ln -sf /opt/conda/bin/python /usr/local/bin/python; \
    elif command -v python3 >/dev/null 2>&1; then ln -sf "$(command -v python3)" /usr/local/bin/python; \
    else echo "ERROR: no python in image" >&2; exit 1; fi; \
    command -v python

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GEMMA_MODEL_PATH=/model \
    GEMMA_IMAGES_DIR=/images \
    GEMMA_API_BACKEND=vllm \
    GEMMA_SKIP_HF_MODEL=1 \
    GEMMA_VLLM_TP_SIZE=2 \
    GEMMA_VLLM_MAX_MODEL_LEN=8192 \
    GEMMA_VLLM_GPU_MEMORY_UTILIZATION=0.90

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker-vllm.txt .
RUN pip install --upgrade pip && pip install -r requirements-docker-vllm.txt

COPY api_server.py gemma_vllm.py gemma_prompts.py .

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=420s --retries=2 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=5).read()"

CMD ["python", "-m", "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "5000", "--log-level", "info"]
