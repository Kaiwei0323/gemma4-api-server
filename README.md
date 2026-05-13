## Gemma 4 Local API (Docker + GPU)

Local HTTP API for Gemma 4, implemented in `api_server.py` and designed to load a **local** model directory (no Hub download).

### Serving backends

| Mode | When to use | Docker |
|------|-------------|--------|
| **`GEMMA_API_BACKEND=vllm`** (default in `Dockerfile`) | Fast **`/chat/stream`**, **`/image/stream`**, **`/video/stream`** via vLLM `AsyncLLM` (tensor parallel, Gemma 4–optimized image) | `FROM vllm/vllm-openai:gemma4-cu130` |
| **`GEMMA_API_BACKEND=hf`** | **`transformers`** stack for **`/chat/stream`**, **`/image/stream`**, **`/video/stream`** (SSE) | `Dockerfile.pytorch-hf` |

With **`GEMMA_SKIP_HF_MODEL=1`** (default in the vLLM `Dockerfile`), the Hugging Face model is **not** loaded, so **`/image/stream`** and **`/video/stream`** need the HF stack unless you set **`GEMMA_SKIP_HF_MODEL=0`** (loads both engines; needs enough VRAM). **`/chat/stream`** uses vLLM when it is loaded.

### Endpoints

- **`POST /chat/stream`**: text → text (**SSE**; JSON or `multipart/form-data`)
- **`POST /image/stream`**: image + text → text (**SSE**; multimodal checkpoint required)
- **`POST /video/stream`**: video + text → text (**SSE**; multimodal checkpoint required)
- **`GET /health`**: model/GPU status
- **`GET /`**: short service metadata

## Prerequisites

- **Docker + GPU (Linux)**: install **NVIDIA Container Toolkit** so the container can use the GPU.
- **Local model folder**: full checkpoint directory, e.g. `./gemma-4-31B-it/` (must include `config.json` and weight shards). This is bind-mounted to `/model`.

## Configuration (`.env`)

```cmd
copy .env.example .env
```

### Key settings

- **`HOST_PORT`**: host port exposed on your machine (default `5000`)
- **`BIND_IP`**:
  - `0.0.0.0` = reachable from LAN (if firewall allows)
  - `127.0.0.1` = localhost only
- **`HOST_MODEL_DIR`**: host path to your model (default `./gemma-4-31B-it`)
- **`HOST_IMAGES_DIR`**: host path to your images folder (default `./images`)
- **`GEMMA_DEVICE_MAP`**:
  - `cuda0` = force all weights on GPU 0 (may OOM)
  - `auto` = Allow distribution across both GPUs
- **`GEMMA_API_KEY`** (optional): if set, requests must include `X-API-Key: ...`
- **`GEMMA_DEFAULT_SYSTEM_PROMPT`** (optional): if set, prepended as the **first** `system` message on **every** **`/chat/stream`**, **`/image/stream`**, and **`/video/stream`** request (each POST), so the model always sees it for that call. Set **`GEMMA_DEFAULT_SYSTEM_RESPECT_CLIENT_SYSTEM=1`** to restore legacy behavior (do not prepend if the client already sent a `system` turn). **`GEMMA_DEFAULT_SYSTEM_FORCE_PREPEND=1`** forces prepend even when `RESPECT` is on.
- **vLLM (when `GEMMA_API_BACKEND=vllm`)**:
  - **`GEMMA_SKIP_HF_MODEL`**: `1` (default in vLLM image) skips loading the in-process HF model (saves VRAM).
  - **`GEMMA_VLLM_TP_SIZE`**: tensor parallel size (default `2` for two GPUs; use `1` on a single GPU).
  - **`GEMMA_VLLM_MAX_MODEL_LEN`**: context length cap (default `8192`).
  - **`GEMMA_VLLM_GPU_MEMORY_UTILIZATION`**: fraction of GPU memory for vLLM (default `0.90`).
  - **`GEMMA_VLLM_DEVICE`**: force vLLM device (`cuda` or `cpu`). Default auto-resolves to `cuda` if available, else `cpu`.
- **Performance tuning (HF path only, optional)**:
  - **`GEMMA_ENABLE_TF32`**: `1` (default) enables TF32 on CUDA for faster matmuls on modern GPUs (minor numeric differences vs full FP32).
  - **`GEMMA_ENABLE_SDPA`**: `1` (default) enables PyTorch scaled-dot-product attention fast paths when available.
  - **`GEMMA_DISABLE_MATH_SDPA`**: `0` (default). Set to `1` only if you know your model shape supports fused SDPA kernels. If set incorrectly, you may get `RuntimeError: No available kernel`.
  - **`GEMMA_USE_AUTOCAST`**: `1` (default) enables CUDA autocast during generation to use bf16/fp16 kernels.
  - **`GEMMA_AUTOCAST_DTYPE`**: `bf16` (default) or `fp16`.
  - **`GEMMA_PERF_LOG`**: `0` (default). Set to `1` to log per-endpoint timing breakdowns (prompt/template time, tokenize time, device transfer, and streaming TTFT).

## Build & run (Docker)

From the folder containing `docker-compose.yml`:

```bash
docker compose --env-file .env up --build -d
docker compose --env-file .env logs -f
```

## Health check (confirm GPU)

```bash
curl -s http://99.64.152.85:5000/health
```

## API examples

These routes stream **incremental text** as Server-Sent Events (SSE). In the browser you usually consume them using `fetch()` + `ReadableStream` (not `EventSource`, because `EventSource` only supports GET).

### `POST /chat/stream`

```bash
curl -N -X POST "http://99.64.152.85:5000/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Write a short poem about rain."}],"max_new_tokens":128}'
```

```bash
curl -N -X POST "http://99.64.152.85:5000/chat/stream" \
  -F "text=Reply with exactly three words." \
  -F "max_new_tokens=64"
```

### `POST /image/stream`

```cmd
curl -N -X POST "http://99.64.152.85:5000/image/stream" ^
  -F "text=Describe this." ^
  -F "image_file=@images/eiffel_tower.jpg" ^
  -F "max_new_tokens=128"
```

### `POST /video/stream`

```cmd
curl -N -X POST "http://99.64.152.85:5000/video/stream" ^
  -F "text=Describe this video." ^
  -F "video_url=@videos/ForBiggerBlazes.mp4" ^
  -F "max_new_tokens=256"
```

## Streaming response shape & latency metrics

All three streaming endpoints emit Server-Sent Events with the same event types:

```text
event: meta    # one-time, includes model_path / backend / default_system_prompt_applied
event: token   # repeated, each delta: {"text": "..."}
event: done    # one-time, final summary (see fields below)
event: error   # only on failure: {"detail": "..."}
```

The final **`event: done`** payload includes per-request latency metrics (all in **seconds**), measured uniformly on both backends (`hf` and `vllm`):

| Field | Meaning |
|------|---------|
| `time_to_first_token_seconds` | Wall time from request start to the **first** decoded output token (TTFT). |
| `prompt_processing_latency_seconds` | Prefill phase — alias of TTFT (template + tokenize + device transfer + prefill + first decode). |
| `decode_latency_seconds` | Total time spent producing tokens **after** the first one (decode phase only). |
| `decode_latency_per_token_seconds` | Mean per-token decode latency: `decode_latency_seconds / max(1, output_tokens − 1)`. `null` when only one token was produced. |
| `end_to_end_latency_seconds` | Full wall time from request start to last token. |
| `tokens_per_second` | `output_tokens / end_to_end_latency_seconds`. |
| `chars_per_second` | Decoded char throughput over end-to-end latency. |
| `output_tokens` | Number of generated tokens. |

Example final event:

```json
{
  "parsed": "...",
  "model_path": "/model",
  "time_to_first_token_seconds": 0.412,
  "prompt_processing_latency_seconds": 0.412,
  "decode_latency_seconds": 1.873,
  "decode_latency_per_token_seconds": 0.0149,
  "end_to_end_latency_seconds": 2.285,
  "tokens_per_second": 56.4,
  "chars_per_second": 271.6,
  "output_tokens": 128
}
```
