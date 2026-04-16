## Gemma 4 Local API (Docker + GPU)

Local HTTP API for Gemma 4, implemented in `api_server.py` and designed to load a **local** model directory (no Hub download).

### Endpoints

- **`POST /chat`**: text → text (JSON or `multipart/form-data`)
- **`POST /chat/stream`**: text → text (**streaming tokens** via SSE over HTTP)
- **`POST /image`**: image + text → text (multimodal checkpoint required)
- **`POST /video`**: video + text → text (multimodal checkpoint required)
- **`POST /audio`**: audio + text → text (multimodal checkpoint required)
- **`GET /health`**: model/GPU status

The **gemma-4-31B-it** checkpoint does not support audio (`POST /audio`).

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

### `POST /chat`

```bash
curl -sS -X POST "http://99.64.152.85:5000/chat" \
  -F "text=Reply with exactly three words." \
  -F "max_new_tokens=64"
```

```bash
curl -sS -X POST "http://99.64.152.85:5000/chat" \
  -F 'messages=[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"Reply with exactly three words."}]' \
  -F "max_new_tokens=64"
```

### `POST /chat/stream` (stream tokens)

This endpoint streams **incremental text** as Server-Sent Events (SSE). In the browser you usually consume it using `fetch()` + `ReadableStream` (not `EventSource`, because `EventSource` only supports GET).

```bash
curl -N -X POST "http://99.64.152.85:5000/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Write a short poem about rain."}],"max_new_tokens":128}'
```

### `POST /image`

```bash
curl -sS -X POST "http://99.64.152.85:5000/image"   -F "text=Describe this."   -F "image_file=@images/eiffel_tower.jpg"   -F "max_new_tokens=128"
```

### `POST /video`

```bash
curl -sS -X POST "http://99.64.152.85:5000/video" \
  -F "text=Describe this video." \
  -F "video_url=@videos/ForBiggerBlazes.mp4" \
  -F "max_new_tokens=256"
```

### `POST /audio`

```bash
curl -sS -X POST "http://99.64.152.85:5000/audio" \
  -F "text=Transcribe the following speech segment in its original language. Only output the transcription, with no newlines." \
  -F "audio_url=@audios/Demos_sample-data_journal1.wav" \
  -F "max_new_tokens=256"
```
