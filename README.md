## Gemma 4 Local API (Docker + GPU)

Local HTTP API for Gemma 4, implemented in `api_server.py` and designed to load a **local** model directory (no Hub download).

### Endpoints

- **`POST /chat`**: text → text
- **`POST /image`**: image + text → text (multimodal checkpoint required)
- **`POST /video`**: video + text → text (multimodal checkpoint required)
- **`POST /audio`**: audio + text → text (multimodal checkpoint required)
- **`GET /health`**: model/GPU status

## Prerequisites

- **Docker + GPU (Linux)**: install **NVIDIA Container Toolkit** so the container can use the GPU.
- **Local model folder**: full checkpoint directory, e.g. `./gemma-4-E4B-it/` (must include `config.json` and weight shards). This is bind-mounted to `/model`.
- **Local images folder (optional)**: put images under `./images/` to use `image_file` with `/image`. This is bind-mounted to `/images`.

## Configuration (`.env`)

Copy the template and edit values as needed.

### Linux/macOS

```bash
cp .env.example .env
```

### Key settings

- **`HOST_PORT`**: host port exposed on your machine (default `5000`)
- **`BIND_IP`**:
  - `0.0.0.0` = reachable from LAN (if firewall allows)
  - `127.0.0.1` = localhost only
- **`HOST_MODEL_DIR`**: host path to your model (default `./gemma-4-E4B-it`)
- **`HOST_IMAGES_DIR`**: host path to your images folder (default `./images`)
- **`GEMMA_DEVICE_MAP`**:
  - `cuda0` = force all weights on GPU 0 (fastest if it fits; may OOM)
  - `auto` = allow GPU + CPU offload (slower, but can run with less VRAM)
- **`GEMMA_API_KEY`** (optional): if set, requests must include `X-API-Key: ...`

## Build & run (Docker)

From the folder containing `docker-compose.yml`:

```bash
docker compose --env-file .env up --build -d
docker compose --env-file .env logs -f
```

## Health check (confirm GPU)

```bash
curl -s http://127.0.0.1:5000/health
```

Look for:
- `"cuda_available": true`
- `"param_devices": ["cuda:0"]`
- `"model_kind": "multimodal"` (needed for `/chat`, `/image`, `/video`, `/audio`)

## API examples

Replace the base URL as needed:
- local: `http://127.0.0.1:5000`
- LAN/WAN: `http://YOUR_HOST:5000`

### `POST /chat`

```bash
curl -sS -X POST "http://99.64.152.85:5000/chat" \
  -H "Content-Type: application/json" \
  --data-binary "@test/chat_test.json"
```

### `POST /image`

```bash
curl -sS -X POST "http://99.64.152.85:5000/image" \
  -H "Content-Type: application/json" \
  --data-binary "@test/image_test.json"
```

### `POST /video`

```bash
curl -sS -X POST "http://99.64.152.85:5000/video" \
  -H "Content-Type: application/json" \
  --data-binary "@test/video_test.json"
```

### `POST /audio`

```bash
curl -sS -X POST "http://99.64.152.85:5000/audio" \
  -H "Content-Type: application/json" \
  --data-binary "@test/audio_test.json"
```

## Common issues

### `/image` / `/video` / `/audio` returns 503 about multimodal

Your checkpoint must load as a multimodal model. `/health` shows `"model_kind": "multimodal"` when this is OK.

### Remote URLs fail inside Docker

Media URLs are fetched **from inside the container**. Avoid `localhost` / `127.*` URLs. Prefer:
- `image_file` under the mounted `./images` folder (for images)
- A host/LAN reachable URL (or `host.docker.internal` on Docker Desktop)

### `from_pretrained` looks stuck in Docker logs

Model load can take a long time. The server prints:
- `STARTUP weights_progress ...` (periodic)
- `STARTUP model_weights_heartbeat ...` (every `GEMMA_LOAD_HEARTBEAT_SEC`)

