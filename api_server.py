"""
HTTP API for local Gemma 4: text chat (`/chat`) and image+text (`/image`, multimodal checkpoint).

Run (recommended: project venv avoids C:\\Python311 permission errors):
  python -m venv .venv
  .\\.venv\\Scripts\\pip install -r requirements.txt
  .\\.venv\\Scripts\\python -m uvicorn api_server:app --host 0.0.0.0 --port 5000

Or: .\\run_server.ps1

If you must use the system Python instead: pip install --user -r requirements.txt
then: python -m uvicorn api_server:app --host 0.0.0.0 --port 5000

Optional auth:
  set GEMMA_API_KEY=your-secret
  Clients must send header: X-API-Key: your-secret

GPU (default: require CUDA and at least some weights on GPU):
  GEMMA_REQUIRE_GPU=1   (default) — refuse startup if CUDA missing or no weights on GPU
  GEMMA_REQUIRE_GPU=0   — allow CPU-only PyTorch (not recommended for Gemma 4)
  GEMMA_DEVICE_MAP=auto (default) — Hugging Face places layers on GPU first, may offload to CPU if VRAM is tight
  GEMMA_DEVICE_MAP=cuda0 — pin entire model to GPU 0 (faster if it fits; may OOM on 12GB)
  GEMMA_MAX_MEMORY_GPU=10GiB — with device_map=auto, cap VRAM used for weights on GPU 0 (raise toward free VRAM to reduce CPU offload; leave ~1–2GiB headroom on 12GB cards)
  GEMMA_MAX_MEMORY_CPU=48GiB — optional cap for CPU weight offload (default 245GiB if unset)
  GEMMA_LOAD_4BIT=1 — quantize weights to ~4 bits so the full model is more likely to fit in VRAM (requires: pip install bitsandbytes)
  GEMMA_LOAD_8BIT=1 — 8-bit quantization (smaller VRAM savings than 4-bit; do not set both 4 and 8)
  GEMMA_LOG_CONFIGURE=1 — (default) reconfigure root logging so STARTUP lines show in `docker compose logs`
  GEMMA_WEIGHTS_TQDM=1 — (default) pass a tqdm class so weight-load progress prints as STARTUP weights_progress lines
  GEMMA_LOAD_HEARTBEAT=1 — (default) print STARTUP model_weights_heartbeat every GEMMA_LOAD_HEARTBEAT_SEC s during from_pretrained
  GEMMA_LOAD_HEARTBEAT_SEC=10 — heartbeat interval (seconds)
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import anyio
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


def _configure_logging_for_docker() -> None:
    """Ensure INFO logs from this module reach stdout (docker compose logs)."""
    fmt = "%(levelname)s [%(name)s] %(message)s"
    root = logging.getLogger()
    _lc = os.environ.get("GEMMA_LOG_CONFIGURE", "1").strip().lower()
    if _lc not in ("0", "false", "no", "off"):
        logging.basicConfig(level=logging.INFO, format=fmt, force=True)
    root.setLevel(logging.INFO)
    for name in ("api_server", "uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).setLevel(logging.INFO)
    logging.getLogger("transformers").setLevel(logging.WARNING)


def _startup_echo(msg: str) -> None:
    """Always visible in `docker compose logs` even if logging handlers are odd."""
    log.info("STARTUP %s", msg)
    print(f"STARTUP {msg}", flush=True)


def _docker_weights_tqdm_class() -> type:
    """tqdm subclass: emit newline STARTUP weights_progress lines (Docker logs are not TTY)."""
    from tqdm import tqdm

    class ComposeLogTqdm(tqdm):
        _last_emit_n = -1

        def __init__(self, *args, **kwargs):
            kwargs = dict(kwargs)
            kwargs.setdefault("file", sys.stdout)
            kwargs.setdefault("mininterval", 2.0)
            kwargs.setdefault("dynamic_ncols", False)
            kwargs.setdefault("ascii", True)
            super().__init__(*args, **kwargs)

        def update(self, n=1) -> bool | None:
            r = super().update(n)
            total = self.total
            desc = (self.desc or "").strip() or "weights"
            if total and total > 0:
                step = max(1, int(total * 0.03))
                if self.n >= total or (self.n - self._last_emit_n >= step):
                    self._last_emit_n = self.n
                    pct = 100.0 * self.n / float(total)
                    print(
                        f"STARTUP weights_progress desc={desc!r} {self.n}/{int(total)} ({pct:.1f}%)",
                        flush=True,
                    )
            elif self.n > 0 and self.n % 400 == 0:
                print(f"STARTUP weights_progress desc={desc!r} n={self.n} (total unknown)", flush=True)
            return r

    return ComposeLogTqdm


def _run_with_heartbeat(label: str, fn: Any, interval_s: float) -> Any:
    """Run fn() while printing elapsed heartbeat lines until fn returns."""
    stop = threading.Event()
    t0 = time.perf_counter()

    def _beat() -> None:
        while not stop.wait(timeout=interval_s):
            print(
                f"STARTUP model_weights_heartbeat label={label} elapsed_s={time.perf_counter() - t0:.0f} "
                f"(from_pretrained still running)",
                flush=True,
            )

    th = threading.Thread(target=_beat, name="load-heartbeat", daemon=True)
    th.start()
    try:
        return fn()
    finally:
        stop.set()
        th.join(timeout=2.0)


_model = None
_processor = None
_model_path: Path | None = None
_load_error: str | None = None
_model_kind: str | None = None  # "multimodal" | "causal" — set after successful load
_gen_lock = threading.Lock()


def _optional_max_memory_for_auto() -> dict[Any, str] | None:
    """Accelerate `max_memory` to bias more layers onto GPU 0 (only used with device_map=auto)."""
    gpu = os.environ.get("GEMMA_MAX_MEMORY_GPU", "").strip()
    if not gpu:
        return None
    cpu = os.environ.get("GEMMA_MAX_MEMORY_CPU", "").strip() or "245GiB"
    return {0: gpu, "cpu": cpu}


def _truthy_env(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _count_param_devices(model: Any) -> tuple[int, int, list[str]]:
    """Returns (num_tensors_on_cuda, total_param_tensors, sorted_unique_devices)."""
    on_cuda = 0
    total = 0
    devices: set[str] = set()
    for p in model.parameters():
        total += 1
        devices.add(str(p.device))
        if p.device.type == "cuda":
            on_cuda += 1
    return on_cuda, total, sorted(devices)


def _default_model_dir() -> Path:
    return Path(__file__).resolve().parent / "gemma-4-E4B-it"


def _default_images_dir() -> Path:
    raw = os.environ.get("GEMMA_IMAGES_DIR", "").strip()
    return Path(raw).expanduser().resolve() if raw else (Path(__file__).resolve().parent / "images")


def _resolve_image_under_images_dir(relative_name: str) -> Path:
    """Resolve a file path under the images directory; reject traversal."""
    if not relative_name or relative_name.strip() != relative_name:
        raise HTTPException(status_code=400, detail="Invalid image_file")
    rel = Path(relative_name)
    if rel.is_absolute():
        raise HTTPException(status_code=400, detail="image_file must be a relative path under the images folder")
    if ".." in rel.parts:
        raise HTTPException(status_code=400, detail="image_file cannot contain '..'")
    base = _default_images_dir().resolve()
    candidate = (base / rel).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        raise HTTPException(status_code=400, detail="image_file must resolve under the images directory")
    if not candidate.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Image not found: {candidate} (images root: {base})",
        )
    return candidate


def _image_content_block(req: ImageRequest) -> dict[str, str]:
    u = (req.image_url or "").strip()
    f = (req.image_file or "").strip()
    if bool(u) == bool(f):
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of: image_url (remote) or image_file (under the images folder).",
        )
    if u:
        try:
            pu = urlparse(u)
        except Exception:
            pu = None
        host = (pu.hostname or "").lower() if pu else ""
        if host in ("localhost",) or host.startswith("127.") or host in ("0.0.0.0",):
            raise HTTPException(
                status_code=400,
                detail=(
                    "image_url points to localhost. If you're running in Docker, localhost refers to the container, "
                    "not your host machine. Use image_file (recommended) or use a host-reachable URL (e.g. "
                    "http://host.docker.internal:PORT/... on Docker Desktop)."
                ),
            )
        return {"type": "image", "url": u}
    path = _resolve_image_under_images_dir(f)
    return {"type": "image", "path": str(path)}


def _validate_remote_media_url(url: str, *, kind: str) -> None:
    """Reject URLs that won't work from inside Docker (e.g. localhost)."""
    try:
        pu = urlparse(url)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid {kind}_url")
    if pu.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail=f"{kind}_url must be http(s)")
    host = (pu.hostname or "").lower()
    if host in ("localhost",) or host.startswith("127.") or host in ("0.0.0.0",):
        raise HTTPException(
            status_code=400,
            detail=(
                f"{kind}_url points to localhost. If you're running in Docker, localhost refers to the container, "
                "not your host machine. Use a host-reachable URL or mount files into the container when possible."
            ),
        )


def _load_model() -> None:
    global _model, _processor, _model_path, _load_error, _model_kind
    _model_kind = None
    t0 = time.perf_counter()
    raw = os.environ.get("GEMMA_MODEL_PATH", "").strip()
    path = Path(raw) if raw else _default_model_dir()
    path = path.expanduser().resolve()
    _startup_echo(f"stage=resolve_model_path path={path}")
    log.info("Model load: resolving path -> %s", path)
    if not path.is_dir():
        _load_error = f"Model directory not found: {path}"
        log.error(_load_error)
        return
    if not (path / "config.json").is_file():
        _load_error = f"No config.json in {path}"
        log.error(_load_error)
        return

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoProcessor
    except ImportError as e:
        _load_error = f"transformers import failed: {e}"
        log.exception("transformers import failed")
        return

    require_gpu = _truthy_env("GEMMA_REQUIRE_GPU", True)
    if require_gpu and not torch.cuda.is_available():
        _load_error = (
            "GEMMA_REQUIRE_GPU is enabled (default) but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled PyTorch wheel and NVIDIA drivers: https://pytorch.org/get-started/locally/ "
            "Or set GEMMA_REQUIRE_GPU=0 to attempt CPU (often impractical for Gemma 4)."
        )
        log.error(_load_error)
        return

    if torch.cuda.is_available():
        _startup_echo(f"stage=cuda_ok device={torch.cuda.get_device_name(0)!r}")
        log.info(
            "Model load: CUDA device 0 = %s",
            torch.cuda.get_device_name(0),
        )
    else:
        _startup_echo("stage=cuda_missing (GEMMA_REQUIRE_GPU may abort)")

    _startup_echo("stage=processor_load_begin")
    log.info("Model load: loading processor (local_files_only=True)...")
    try:
        processor = AutoProcessor.from_pretrained(str(path), local_files_only=True, padding_side="left")
    except Exception as e:
        _load_error = f"Failed to load processor: {e}"
        log.exception("Failed to load processor")
        return
    _startup_echo(f"stage=processor_load_done elapsed_s={time.perf_counter() - t0:.1f}")
    log.info("Model load: processor ready in %.1fs", time.perf_counter() - t0)

    device_map_raw = os.environ.get("GEMMA_DEVICE_MAP", "auto").strip().lower()
    if device_map_raw in ("0", "cuda", "cuda:0", "cuda0", "gpu", "gpu0"):
        device_map: Any = {"": 0}
        log.info("Model load: GEMMA_DEVICE_MAP pins full model to cuda:0 (may OOM if VRAM is insufficient)")
    elif device_map_raw in ("", "auto"):
        device_map = "auto"
    else:
        device_map = os.environ.get("GEMMA_DEVICE_MAP", "auto").strip() or "auto"
        log.info("Model load: using device_map=%r", device_map)

    load_kw: dict[str, Any] = {
        "dtype": "auto",
        "device_map": device_map,
        "local_files_only": True,
    }

    use_4bit = _truthy_env("GEMMA_LOAD_4BIT", False)
    use_8bit = _truthy_env("GEMMA_LOAD_8BIT", False)
    if use_4bit and use_8bit:
        _load_error = "Set only one of GEMMA_LOAD_4BIT or GEMMA_LOAD_8BIT"
        log.error(_load_error)
        return
    if use_4bit or use_8bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as e:
            _load_error = f"GEMMA_LOAD_4BIT/8BIT requires transformers BitsAndBytesConfig: {e}"
            log.exception(_load_error)
            return
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            _load_error = "GEMMA_LOAD_4BIT/8BIT requires: pip install bitsandbytes (Windows: use a CUDA build; see bitsandbytes docs)"
            log.error(_load_error)
            return
        if use_4bit:
            load_kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            log.info("Model load: GEMMA_LOAD_4BIT=1 (NF4 weights, bf16 compute)")
        else:
            load_kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            log.info("Model load: GEMMA_LOAD_8BIT=1")
        load_kw.pop("dtype", None)

    if device_map == "auto" and not use_4bit and not use_8bit:
        mm = _optional_max_memory_for_auto()
        if mm is not None:
            load_kw["max_memory"] = mm
            log.info("Model load: max_memory=%r (larger GEMMA_MAX_MEMORY_GPU => fewer layers on CPU if they fit)", mm)

    attn = os.environ.get("GEMMA_ATTN_IMPLEMENTATION", "").strip()
    if attn:
        load_kw["attn_implementation"] = attn

    _startup_echo(
        f"stage=model_weights_begin device_map={load_kw.get('device_map')!r} "
        f"4bit={use_4bit} 8bit={use_8bit} — from_pretrained can take many minutes"
    )
    model = None
    load_class: type | None = None
    last_err: Exception | None = None
    # Prefer multimodal first so /image works when the checkpoint supports it.
    for cls in (AutoModelForMultimodalLM, AutoModelForCausalLM):
        log.info("Model load: trying %s.from_pretrained (this can take several minutes)...", cls.__name__)
        _startup_echo(f"stage=model_from_pretrained_begin class={cls.__name__}")
        t_cls = time.perf_counter()
        try:
            fp_kw = dict(load_kw)
            if _truthy_env("GEMMA_WEIGHTS_TQDM", True):
                try:
                    fp_kw["tqdm_class"] = _docker_weights_tqdm_class()
                except Exception as e:
                    log.warning("Could not enable Docker tqdm for weights: %s", e)

            def _do_load() -> Any:
                return cls.from_pretrained(str(path), **fp_kw)

            try:
                hb_sec = float(os.environ.get("GEMMA_LOAD_HEARTBEAT_SEC", "10").strip() or "10")
            except ValueError:
                hb_sec = 10.0
            if _truthy_env("GEMMA_LOAD_HEARTBEAT", True) and hb_sec > 0:
                model = _run_with_heartbeat(cls.__name__, _do_load, hb_sec)
            else:
                model = _do_load()
            load_class = cls
            try:
                p0 = next(model.parameters())
                dev_info = str(p0.device)
                dtype_info = str(p0.dtype)
            except Exception:
                dev_info, dtype_info = "unknown", "unknown"
            log.info(
                "Model load: %s succeeded in %.1fs; param_device=%s param_dtype=%s",
                cls.__name__,
                time.perf_counter() - t_cls,
                dev_info,
                dtype_info,
            )
            _startup_echo(
                f"stage=model_class_loaded class={cls.__name__} "
                f"wall_s={time.perf_counter() - t_cls:.1f} param_device={dev_info}"
            )
            break
        except Exception as e:
            last_err = e
            _startup_echo(f"stage=model_from_pretrained_failed class={cls.__name__} err={e!r}")
            log.warning("Model load: %s failed: %s", cls.__name__, e)
    if model is None or load_class is None:
        _load_error = f"Failed to load model with AutoModel classes: {last_err}"
        log.error(_load_error)
        return

    _model_kind = "multimodal" if load_class is AutoModelForMultimodalLM else "causal"

    on_cuda, param_tensors, devices = _count_param_devices(model)
    _startup_echo(
        f"stage=param_device_check on_cuda={on_cuda} total_tensors={param_tensors} devices={devices!r}"
    )
    log.info(
        "Model load: parameter tensors on CUDA=%s / total=%s devices_sample=%s",
        on_cuda,
        param_tensors,
        devices[:6] if len(devices) > 6 else devices,
    )
    if require_gpu and on_cuda == 0:
        _load_error = (
            "GEMMA_REQUIRE_GPU is set but no model parameters were placed on CUDA (all on CPU). "
            "Try GEMMA_DEVICE_MAP=cuda0 if the model fits in VRAM, or verify torch GPU install."
        )
        log.error(_load_error)
        return

    _processor = processor
    _model = model
    _model_path = path
    _load_error = None
    _startup_echo(
        f"stage=model_load_complete total_s={time.perf_counter() - t0:.1f} model_kind={_model_kind!r} — HTTP API ready"
    )
    log.info(
        "Model load: complete in %.1fs — ready for /chat (%s)",
        time.perf_counter() - t0,
        _model_kind,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_logging_for_docker()
    _startup_echo(
        "stage=uvicorn_lifespan_enter — next: load processor + model (see STARTUP lines; this blocks until done)"
    )
    log.info(
        "Startup: pid=%s main_thread=%s cwd=%s",
        os.getpid(),
        threading.main_thread().name,
        os.getcwd(),
    )
    _load_model()
    if _load_error:
        _startup_echo(f"stage=lifespan_model_failed error={_load_error!r}")
        log.warning("Startup finished with model error (service degraded): %s", _load_error)
    else:
        _startup_echo("stage=lifespan_model_ok — uvicorn will print 'Application startup complete'")
        log.info("Startup: application ready (uvicorn will log 'Application startup complete')")
    yield


app = FastAPI(title="Gemma 4 local chat API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("GEMMA_CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.environ.get("GEMMA_API_KEY", "").strip()
    if not expected:
        return
    if not x_api_key or x_api_key.strip() != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


class ChatMessage(BaseModel):
    role: str = Field(..., description="system, user, or assistant")
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)
    enable_thinking: bool = False
    max_new_tokens: int = Field(1024, ge=1, le=8192)
    temperature: float | None = Field(None, ge=0, le=2)
    top_p: float | None = Field(None, ge=0, le=1)
    top_k: int | None = Field(None, ge=1)
    do_sample: bool | None = None
    include_raw: bool = Field(False, description="If true, include raw decoded assistant string")


class ChatResponse(BaseModel):
    parsed: Any
    raw: str | None = None
    model_path: str | None = None
    tokens_per_second: float | None = Field(
        default=None,
        description="New tokens per second during model.generate() only (output length − prompt length, wall clock).",
    )


def _new_token_count_and_tps(outputs: Any, input_len: int, elapsed_s: float) -> tuple[int, float | None]:
    n_new = max(0, int(outputs[0].shape[-1]) - int(input_len))
    if elapsed_s <= 1e-9:
        return n_new, None
    return n_new, round(n_new / elapsed_s, 3)


class ImageRequest(BaseModel):
    image_url: str | None = Field(
        default=None,
        description="HTTP(S) URL of the image (use this or image_file, not both)",
    )
    image_file: str | None = Field(
        default=None,
        description="Filename or subpath under the `images` folder, e.g. `photo.png` or `shots/frame.jpg`",
    )
    text: str = Field(..., description="Question or instruction about the image")
    enable_thinking: bool = False
    max_new_tokens: int = Field(512, ge=1, le=8192)
    temperature: float | None = Field(None, ge=0, le=2)
    top_p: float | None = Field(None, ge=0, le=1)
    top_k: int | None = Field(None, ge=1)
    do_sample: bool | None = None
    include_raw: bool = False


class VideoRequest(BaseModel):
    video_url: str = Field(..., description="HTTP(S) URL of the video (mp4/webm).")
    text: str = Field(..., description="Question or instruction about the video")
    enable_thinking: bool = False
    max_new_tokens: int = Field(512, ge=1, le=8192)
    temperature: float | None = Field(None, ge=0, le=2)
    top_p: float | None = Field(None, ge=0, le=1)
    top_k: int | None = Field(None, ge=1)
    do_sample: bool | None = None
    include_raw: bool = False


class AudioRequest(BaseModel):
    audio_url: str = Field(..., description="HTTP(S) URL of the audio (wav/mp3/flac).")
    text: str = Field(..., description="Question or instruction about the audio")
    enable_thinking: bool = False
    max_new_tokens: int = Field(512, ge=1, le=8192)
    temperature: float | None = Field(None, ge=0, le=2)
    top_p: float | None = Field(None, ge=0, le=1)
    top_k: int | None = Field(None, ge=1)
    do_sample: bool | None = None
    include_raw: bool = False


def _inference_device() -> Any:
    assert _model is not None
    dev = getattr(_model, "device", None)
    if dev is not None:
        return dev
    for p in _model.parameters():
        if p.device.type == "cuda":
            return p.device
    return next(_model.parameters()).device


def _build_generation_kwargs(
    max_new_tokens: int,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    do_sample: bool | None,
) -> dict[str, Any]:
    gen_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
    resolved_sample = do_sample
    if resolved_sample is None:
        resolved_sample = temperature is not None and temperature > 0
    gen_kwargs["do_sample"] = resolved_sample
    if resolved_sample:
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        if top_k is not None:
            gen_kwargs["top_k"] = top_k
    else:
        gen_kwargs["top_p"] = 1.0
        gen_kwargs["top_k"] = 50
    return gen_kwargs


def _generate_sync(req: ChatRequest) -> ChatResponse:
    assert _model is not None and _processor is not None

    messages = [m.model_dump() for m in req.messages]

    text = _processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=req.enable_thinking,
    )
    inputs = _processor(text=text, return_tensors="pt").to(_inference_device())
    input_len = inputs["input_ids"].shape[-1]

    gen_kwargs = _build_generation_kwargs(
        req.max_new_tokens,
        req.temperature,
        req.top_p,
        req.top_k,
        req.do_sample,
    )

    log.info(
        "Chat: generate max_new_tokens=%s do_sample=%s input_len=%s",
        req.max_new_tokens,
        gen_kwargs["do_sample"],
        input_len,
    )
    t_gen0 = time.perf_counter()
    with _gen_lock:
        outputs = _model.generate(**inputs, **gen_kwargs)
    gen_elapsed = time.perf_counter() - t_gen0
    _, tps = _new_token_count_and_tps(outputs, input_len, gen_elapsed)
    log.info("Chat: generate wall=%.3fs tokens_per_second=%s", gen_elapsed, tps)

    response = _processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    parsed = _processor.parse_response(response)

    return ChatResponse(
        parsed=parsed,
        raw=response if req.include_raw else None,
        model_path=str(_model_path) if _model_path else None,
        tokens_per_second=tps,
    )


def _image_generate_sync(req: ImageRequest, image_block: dict[str, str]) -> ChatResponse:
    assert _model is not None and _processor is not None

    messages = [
        {
            "role": "user",
            "content": [
                image_block,
                {"type": "text", "text": req.text},
            ],
        }
    ]
    tpl_kw: dict[str, Any] = {
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "add_generation_prompt": True,
    }
    try:
        inputs = _processor.apply_chat_template(
            messages, **tpl_kw, enable_thinking=req.enable_thinking
        )
    except TypeError:
        inputs = _processor.apply_chat_template(messages, **tpl_kw)

    inputs = inputs.to(_inference_device())
    input_len = inputs["input_ids"].shape[-1]

    gen_kwargs = _build_generation_kwargs(
        req.max_new_tokens,
        req.temperature,
        req.top_p,
        req.top_k,
        req.do_sample,
    )

    src = image_block.get("path") or image_block.get("url", "")
    log.info(
        "Image: generate src=%s max_new_tokens=%s do_sample=%s input_len=%s",
        src[:120] + ("..." if len(src) > 120 else ""),
        req.max_new_tokens,
        gen_kwargs["do_sample"],
        input_len,
    )
    t_gen0 = time.perf_counter()
    with _gen_lock:
        outputs = _model.generate(**inputs, **gen_kwargs)
    gen_elapsed = time.perf_counter() - t_gen0
    _, tps = _new_token_count_and_tps(outputs, input_len, gen_elapsed)
    log.info("Image: generate wall=%.3fs tokens_per_second=%s", gen_elapsed, tps)

    response = _processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    parsed = _processor.parse_response(response)

    return ChatResponse(
        parsed=parsed,
        raw=response if req.include_raw else None,
        model_path=str(_model_path) if _model_path else None,
        tokens_per_second=tps,
    )


def _video_generate_sync(req: VideoRequest) -> ChatResponse:
    assert _model is not None and _processor is not None

    video_url = (req.video_url or "").strip()
    if not video_url:
        raise HTTPException(status_code=400, detail="video_url is required")
    _validate_remote_media_url(video_url, kind="video")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_url},
                {"type": "text", "text": req.text},
            ],
        }
    ]
    tpl_kw: dict[str, Any] = {
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "add_generation_prompt": True,
    }
    try:
        try:
            inputs = _processor.apply_chat_template(
                messages, **tpl_kw, enable_thinking=req.enable_thinking
            )
        except TypeError:
            inputs = _processor.apply_chat_template(messages, **tpl_kw)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Video: processor/apply_chat_template failed")
        raise HTTPException(
            status_code=502,
            detail=(
                "Failed to fetch/process video_url inside the server. "
                f"{e.__class__.__name__}: {e}"
            ),
        )

    inputs = inputs.to(_inference_device())
    input_len = inputs["input_ids"].shape[-1]

    gen_kwargs = _build_generation_kwargs(
        req.max_new_tokens,
        req.temperature,
        req.top_p,
        req.top_k,
        req.do_sample,
    )

    log.info(
        "Video: generate url=%s max_new_tokens=%s do_sample=%s input_len=%s",
        video_url[:120] + ("..." if len(video_url) > 120 else ""),
        req.max_new_tokens,
        gen_kwargs["do_sample"],
        input_len,
    )
    t_gen0 = time.perf_counter()
    with _gen_lock:
        outputs = _model.generate(**inputs, **gen_kwargs)
    gen_elapsed = time.perf_counter() - t_gen0
    _, tps = _new_token_count_and_tps(outputs, input_len, gen_elapsed)
    log.info("Video: generate wall=%.3fs tokens_per_second=%s", gen_elapsed, tps)

    response = _processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    parsed = _processor.parse_response(response)

    return ChatResponse(
        parsed=parsed,
        raw=response if req.include_raw else None,
        model_path=str(_model_path) if _model_path else None,
        tokens_per_second=tps,
    )


def _audio_generate_sync(req: AudioRequest) -> ChatResponse:
    assert _model is not None and _processor is not None

    audio_url = (req.audio_url or "").strip()
    if not audio_url:
        raise HTTPException(status_code=400, detail="audio_url is required")
    _validate_remote_media_url(audio_url, kind="audio")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_url},
                {"type": "text", "text": req.text},
            ],
        }
    ]
    tpl_kw: dict[str, Any] = {
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "add_generation_prompt": True,
    }
    try:
        try:
            inputs = _processor.apply_chat_template(
                messages, **tpl_kw, enable_thinking=req.enable_thinking
            )
        except TypeError:
            inputs = _processor.apply_chat_template(messages, **tpl_kw)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Audio: processor/apply_chat_template failed")
        raise HTTPException(
            status_code=502,
            detail=(
                "Failed to fetch/process audio_url inside the server. "
                f"{e.__class__.__name__}: {e}"
            ),
        )

    inputs = inputs.to(_inference_device())
    input_len = inputs["input_ids"].shape[-1]

    gen_kwargs = _build_generation_kwargs(
        req.max_new_tokens,
        req.temperature,
        req.top_p,
        req.top_k,
        req.do_sample,
    )

    log.info(
        "Audio: generate url=%s max_new_tokens=%s do_sample=%s input_len=%s",
        audio_url[:120] + ("..." if len(audio_url) > 120 else ""),
        req.max_new_tokens,
        gen_kwargs["do_sample"],
        input_len,
    )
    t_gen0 = time.perf_counter()
    with _gen_lock:
        outputs = _model.generate(**inputs, **gen_kwargs)
    gen_elapsed = time.perf_counter() - t_gen0
    _, tps = _new_token_count_and_tps(outputs, input_len, gen_elapsed)
    log.info("Audio: generate wall=%.3fs tokens_per_second=%s", gen_elapsed, tps)

    response = _processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    parsed = _processor.parse_response(response)

    return ChatResponse(
        parsed=parsed,
        raw=response if req.include_raw else None,
        model_path=str(_model_path) if _model_path else None,
        tokens_per_second=tps,
    )


@app.get("/health")
def health():
    ok = _model is not None and _processor is not None
    gpu_block: dict[str, Any] = {"require_gpu": _truthy_env("GEMMA_REQUIRE_GPU", True)}
    try:
        import torch

        gpu_block["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            gpu_block["cuda_device_0"] = torch.cuda.get_device_name(0)
        if ok and _model is not None:
            on_cuda, total, devices = _count_param_devices(_model)
            gpu_block["param_tensors_on_cuda"] = on_cuda
            gpu_block["param_tensors_total"] = total
            gpu_block["param_devices"] = devices
    except Exception as e:
        gpu_block["error"] = str(e)
    return {
        "status": "ok" if ok else "degraded",
        "model_loaded": ok,
        "model_kind": _model_kind,
        "image_endpoint": ok and _model_kind == "multimodal",
        "images_dir": str(_default_images_dir()),
        "gpu": gpu_block,
        "error": _load_error,
        "model_path": str(_model_path) if _model_path else None,
        "offload_hints": {
            "GEMMA_DEVICE_MAP": os.environ.get("GEMMA_DEVICE_MAP", "auto"),
            "GEMMA_MAX_MEMORY_GPU": os.environ.get("GEMMA_MAX_MEMORY_GPU", ""),
            "GEMMA_MAX_MEMORY_CPU": os.environ.get("GEMMA_MAX_MEMORY_CPU", ""),
            "GEMMA_LOAD_4BIT": _truthy_env("GEMMA_LOAD_4BIT", False),
            "GEMMA_LOAD_8BIT": _truthy_env("GEMMA_LOAD_8BIT", False),
        },
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, _: None = Depends(require_api_key)):
    if _model is None or _processor is None:
        raise HTTPException(status_code=503, detail=_load_error or "Model not loaded")

    return await anyio.to_thread.run_sync(lambda: _generate_sync(req))


@app.post("/image", response_model=ChatResponse)
async def image_chat(req: ImageRequest, _: None = Depends(require_api_key)):
    if _model is None or _processor is None:
        raise HTTPException(status_code=503, detail=_load_error or "Model not loaded")
    if _model_kind != "multimodal":
        raise HTTPException(
            status_code=503,
            detail=(
                "Image input requires a multimodal checkpoint loaded as AutoModelForMultimodalLM. "
                f"This instance loaded as {_model_kind!r}. Use a Gemma 4 multimodal model folder or check logs."
            ),
        )

    image_block = _image_content_block(req)
    return await anyio.to_thread.run_sync(lambda: _image_generate_sync(req, image_block))


@app.post("/video", response_model=ChatResponse)
async def video_chat(req: VideoRequest, _: None = Depends(require_api_key)):
    if _model is None or _processor is None:
        raise HTTPException(status_code=503, detail=_load_error or "Model not loaded")
    if _model_kind != "multimodal":
        raise HTTPException(
            status_code=503,
            detail=(
                "Video input requires a multimodal checkpoint loaded as AutoModelForMultimodalLM. "
                f"This instance loaded as {_model_kind!r}. Use a Gemma 4 multimodal model folder or check logs."
            ),
        )

    return await anyio.to_thread.run_sync(lambda: _video_generate_sync(req))


@app.post("/audio", response_model=ChatResponse)
async def audio_chat(req: AudioRequest, _: None = Depends(require_api_key)):
    if _model is None or _processor is None:
        raise HTTPException(status_code=503, detail=_load_error or "Model not loaded")
    if _model_kind != "multimodal":
        raise HTTPException(
            status_code=503,
            detail=(
                "Audio input requires a multimodal checkpoint loaded as AutoModelForMultimodalLM. "
                f"This instance loaded as {_model_kind!r}. Use a Gemma 4 multimodal model folder or check logs."
            ),
        )

    return await anyio.to_thread.run_sync(lambda: _audio_generate_sync(req))


@app.get("/")
def root():
    return {
        "service": "gemma-4-local-chat",
        "endpoints": {
            "POST /chat": "text chat",
            "POST /image": "image (image_url or image_file under ./images) + text — multimodal model only",
            "POST /video": "video (video_url) + text — multimodal model only",
            "POST /audio": "audio (audio_url) + text — multimodal model only",
            "GET /health": "load status",
        },
        "model_path_env": "GEMMA_MODEL_PATH (default: ./gemma-4-E4B-it next to api_server.py)",
        "images_dir_env": "GEMMA_IMAGES_DIR (default: ./images next to api_server.py)",
    }
