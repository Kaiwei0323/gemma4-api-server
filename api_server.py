"""
HTTP API for local Gemma 4: streaming only — `POST /chat/stream`, `POST /image/stream`, `POST /video/stream` (SSE).

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
  GEMMA_MAX_MEMORY_GPU=10GiB — with device_map=auto, cap per-GPU VRAM for each visible CUDA device (raise toward free VRAM to reduce CPU offload; leave ~1–2GiB headroom per GPU)
  GEMMA_MAX_MEMORY_CPU=48GiB — optional cap for CPU weight offload (default 245GiB if unset)
  GEMMA_LOAD_4BIT=1 — quantize weights to ~4 bits so the full model is more likely to fit in VRAM (requires: pip install bitsandbytes)
  GEMMA_LOAD_8BIT=1 — 8-bit quantization (smaller VRAM savings than 4-bit; do not set both 4 and 8)
  GEMMA_LOG_CONFIGURE=1 — (default) reconfigure root logging so STARTUP lines show in `docker compose logs`
  GEMMA_WEIGHTS_TQDM=1 — (default) pass a tqdm class so weight-load progress prints as STARTUP weights_progress lines
  GEMMA_LOAD_HEARTBEAT=1 — (default) print STARTUP model_weights_heartbeat every GEMMA_LOAD_HEARTBEAT_SEC s during from_pretrained
  GEMMA_LOAD_HEARTBEAT_SEC=10 — heartbeat interval (seconds)
  Multimodal preprocessing (faster / lower quality — optional):
  GEMMA_VIDEO_NUM_FRAMES=16 — sample fewer frames from each video (default from processor, often 32; lower = faster decode + vision)
  GEMMA_VIDEO_MAX_SOFT_TOKENS=70 — vision tokens per video frame (must be one of 70,140,280,560,1120; lower = faster, rougher)
  GEMMA_IMAGE_MAX_SOFT_TOKENS=280 — same allowed set; lower = faster image understanding, less detail
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
import uuid
import json
from contextlib import ExitStack, asynccontextmanager, nullcontext
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from gemma_prompts import (
    default_system_prompt_text,
    default_system_prompt_would_prepend,
    prepend_default_system,
)

log = logging.getLogger(__name__)

def _perf_enabled() -> bool:
    return _truthy_env("GEMMA_PERF_LOG", False)


def _perf(msg: str, *args: Any) -> None:
    if _perf_enabled():
        log.info("PERF " + msg, *args)


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
_model_supports_audio: bool | None = None  # True if config.audio_config is set (audio tower weights)
_gen_lock = threading.Lock()
_transformers_runtime_patches_applied = False


def _optional_max_memory_for_auto() -> dict[Any, str] | None:
    """Build Accelerate `max_memory` for device_map=auto (only used when GEMMA_MAX_MEMORY_GPU is set).

    The same per-GPU cap is applied to every visible CUDA device so multi-GPU splits (e.g. 31B on 2×48GB)
    stay balanced; previously only device 0 was capped.
    """
    gpu = os.environ.get("GEMMA_MAX_MEMORY_GPU", "").strip()
    if not gpu:
        return None
    cpu = os.environ.get("GEMMA_MAX_MEMORY_CPU", "").strip() or "245GiB"
    try:
        import torch

        n = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        n = 0
    if n <= 0:
        return {0: gpu, "cpu": cpu}
    mm: dict[Any, str] = {i: gpu for i in range(n)}
    mm["cpu"] = cpu
    return mm


def _truthy_env(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _api_backend() -> str:
    """`hf` = Hugging Face `transformers` in-process (default). `vllm` = vLLM AsyncLLM for stream routes."""
    return os.environ.get("GEMMA_API_BACKEND", "hf").strip().lower()


def _vllm_ready() -> bool:
    try:
        import gemma_vllm as gv

        return gv.is_loaded()
    except Exception:
        return False


def _apply_transformers_runtime_patches() -> None:
    """
    Transformers may prefer `torchcodec` for audio decoding when installed, but torchcodec can be
    present yet unusable in minimal Docker images (missing FFmpeg shared libs / ABI mismatch).

    Default: force librosa-based audio decoding unless GEMMA_USE_TORCHCODEC=1 and import works.
    """
    global _transformers_runtime_patches_applied
    if _transformers_runtime_patches_applied:
        return

    try:
        import transformers.audio_utils as audio_utils  # type: ignore
        import transformers.utils.import_utils as import_utils  # type: ignore
    except Exception as e:
        log.warning("Runtime patches: could not import transformers modules: %s", e)
        _transformers_runtime_patches_applied = True
        return

    def _disable_torchcodec_audio() -> None:
        def _no_torchcodec() -> bool:
            return False

        try:
            audio_utils.is_torchcodec_available = _no_torchcodec  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            import_utils.is_torchcodec_available = _no_torchcodec  # type: ignore[attr-defined]
        except Exception:
            pass

    use_tc = _truthy_env("GEMMA_USE_TORCHCODEC", False)
    if not use_tc:
        _disable_torchcodec_audio()
        log.info("Runtime patches: GEMMA_USE_TORCHCODEC disabled — using librosa audio decoding path")
        _transformers_runtime_patches_applied = True
        return

    try:
        import torchcodec  # noqa: F401
    except Exception as e:
        log.warning(
            "Runtime patches: GEMMA_USE_TORCHCODEC=1 but torchcodec import failed (%s); "
            "falling back to librosa audio decoding",
            e,
        )
        _disable_torchcodec_audio()

    _transformers_runtime_patches_applied = True


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


def _default_upload_tmp_dir() -> Path:
    raw = os.environ.get("GEMMA_UPLOAD_TMP_DIR", "").strip()
    base = Path(raw).expanduser() if raw else Path("/tmp/gemma4-uploads")
    base = base.resolve()
    base.mkdir(parents=True, exist_ok=True)
    return base


def _resolve_image_under_images_dir(relative_name: str) -> Path:
    """Resolve a file path under the images directory; reject traversal."""
    if not relative_name or relative_name.strip() != relative_name:
        raise HTTPException(status_code=400, detail="Invalid image_file")
    raw = Path(relative_name)
    if ".." in raw.parts:
        raise HTTPException(status_code=400, detail="image_file cannot contain '..'")
    base = _default_images_dir().resolve()
    candidate = raw.expanduser().resolve() if raw.is_absolute() else (base / raw).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=(
                "image_file must resolve under the configured images directory "
                f"(GEMMA_IMAGES_DIR / images root: {base}). "
                "In Docker, host paths like /home/... are usually not visible inside the container; "
                "use a path under the bind mount (often `/images/...`) or a relative name like `tiger.jpg`."
            ),
        )
    if not candidate.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Image not found: {candidate} (images root: {base})",
        )
    return candidate


def _truthy_form_value(raw: str | None) -> bool | None:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s == "":
        return None
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    raise HTTPException(status_code=400, detail=f"Invalid boolean form value: {raw!r}")


def _parse_optional_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "":
        return None
    try:
        return int(s)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid integer form value: {raw!r}")


def _parse_optional_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid float form value: {raw!r}")


def _is_starlette_upload_file(obj: Any) -> bool:
    """Starlette/FastAPI file uploads duck-type like this; avoid strict isinstance mismatches."""
    return hasattr(obj, "read") and hasattr(obj, "filename")


def _json_safe_validation_detail(obj: Any) -> Any:
    """Make FastAPI validation errors JSON-serializable (avoid bytes in `input`)."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if k == "input":
                if isinstance(v, (bytes, bytearray, memoryview)):
                    out[k] = f"<{type(v).__name__} len={len(v)}>"
                elif _is_starlette_upload_file(v):
                    out[k] = "<upload>"
                else:
                    out[k] = _json_safe_validation_detail(v)
            else:
                out[k] = _json_safe_validation_detail(v)
        return out
    if isinstance(obj, list):
        return [_json_safe_validation_detail(x) for x in obj]
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return f"<{type(obj).__name__} len={len(obj)}>"
    if _is_starlette_upload_file(obj):
        return "<upload>"
    return obj


def _body_looks_like_multipart(body: bytes) -> bool:
    # multipart bodies almost always begin with "--" + boundary (RFC 2046)
    return bool(body) and body.lstrip().startswith(b"--")


def _parse_json_model(model_cls: type[BaseModel], body: bytes, *, route: str) -> BaseModel:
    """
    Parse JSON into a pydantic model without letting validation errors include raw binary `input`
    (which can crash FastAPI's jsonable_encoder on /video and /image).
    """
    if _body_looks_like_multipart(body):
        raise HTTPException(
            status_code=415,
            detail=(
                f"{route}: received multipart body but Content-Type was not multipart/form-data. "
                "Send multipart with `-F ...`, or send JSON with `-H \"Content-Type: application/json\"`."
            ),
        )
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as e:
        raise HTTPException(
            status_code=415,
            detail=(
                f"{route}: body is not UTF-8 JSON. If you intended to upload a file, use multipart/form-data "
                f"(`curl -F ...`). Decode error: {e}"
            ),
        )
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"{route}: invalid JSON: {e}")

    try:
        return model_cls.model_validate(data)
    except ValidationError as e:
        # Sanitize: drop `input` fields that may contain bytes/non-utf8 fragments.
        safe: list[dict[str, Any]] = []
        for err in e.errors():
            e2 = dict(err)
            e2.pop("input", None)
            safe.append(e2)
        raise HTTPException(status_code=422, detail=safe)


async def _image_request_from_multipart(request: Request) -> tuple[ImageRequest, Path | None, str | None]:
    """
    Parse multipart/form-data for /image.

    Typical curl:
      curl -F "text=..." -F "image_file=@tiger.jpg" -F "max_new_tokens=128" ...

    Provide exactly one of:
    - image_url
    - image_file (path string under GEMMA_IMAGES_DIR)
    - one uploaded file part (recommended field name: image_file)
    """
    form = await request.form()

    text = (form.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Missing form field: text")

    image_url = (form.get("image_url") or "").strip() or None

    image_file_field = form.get("image_file")
    image_file: str | None = None
    chosen_upload: UploadFile | None = None
    if _is_starlette_upload_file(image_file_field):
        chosen_upload = image_file_field  # type: ignore[assignment]
    elif image_file_field is None:
        image_file = None
    else:
        image_file = str(image_file_field).strip() or None
        if image_file.startswith("UploadFile("):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Malformed multipart request: `image_file` was not parsed as a file upload. "
                    "Use: -F \"image_file=@your.jpg\" (note the '@')."
                ),
            )

    uploads: list[Any] = []
    for _, v in form.multi_items():
        if _is_starlette_upload_file(v):
            uploads.append(v)

    if chosen_upload is None:
        # Allow a single unnamed upload part (or pick explicitly when multiple parts exist).
        if len(uploads) == 1:
            chosen_upload = uploads[0]  # type: ignore[assignment]
        elif len(uploads) > 1:
            raise HTTPException(
                status_code=400,
                detail="Multiple file parts were uploaded; specify the image part as field name `image_file`.",
            )

    tmp_path: Path | None = None
    uploaded_abs_path: str | None = None
    if chosen_upload is not None:
        if image_url or image_file:
            raise HTTPException(
                status_code=400,
                detail="Provide only one of: image_url, image_file (path string), or a single uploaded image file.",
            )
        up = chosen_upload
        raw_name = (up.filename or "upload").strip() or "upload"
        safe_name = Path(raw_name).name
        suffix = Path(safe_name).suffix.lower()
        allowed = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
        if suffix and suffix not in allowed:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported uploaded image extension {suffix!r}. Allowed: {sorted(allowed)}",
            )

        data = await up.read()
        if not data:
            raise HTTPException(status_code=400, detail="Uploaded image file is empty")

        uploads_dir = _default_upload_tmp_dir()
        tmp_path = uploads_dir / f"{uuid.uuid4().hex}{suffix or '.bin'}"
        tmp_path.write_bytes(data)
        uploaded_abs_path = str(tmp_path.resolve())

    enable_thinking = _truthy_form_value(form.get("enable_thinking"))  # type: ignore[arg-type]
    do_sample = _truthy_form_value(form.get("do_sample"))  # type: ignore[arg-type]
    include_raw = _truthy_form_value(form.get("include_raw"))  # type: ignore[arg-type]
    image_max_soft_tokens = _parse_optional_int(form.get("image_max_soft_tokens"))  # type: ignore[arg-type]

    if uploaded_abs_path is not None:
        # Bypass GEMMA_IMAGES_DIR resolution: multipart uploads are staged under GEMMA_UPLOAD_TMP_DIR (writable).
        req = ImageRequest(
            image_url=None,
            image_file=None,
            image_upload_path=uploaded_abs_path,
            text=text,
            enable_thinking=bool(enable_thinking) if enable_thinking is not None else False,
            max_new_tokens=_parse_optional_int(form.get("max_new_tokens")) or 512,  # type: ignore[arg-type]
            temperature=_parse_optional_float(form.get("temperature")),  # type: ignore[arg-type]
            top_p=_parse_optional_float(form.get("top_p")),  # type: ignore[arg-type]
            top_k=_parse_optional_int(form.get("top_k")),  # type: ignore[arg-type]
            do_sample=do_sample,
            include_raw=bool(include_raw) if include_raw is not None else False,
            image_max_soft_tokens=image_max_soft_tokens,
        )
        return req, tmp_path, uploaded_abs_path

    if bool(image_url) == bool(image_file):
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of: image_url, image_file (path under GEMMA_IMAGES_DIR), or one file upload.",
        )

    req = ImageRequest(
        image_url=image_url,
        image_file=image_file,
        text=text,
        enable_thinking=bool(enable_thinking) if enable_thinking is not None else False,
        max_new_tokens=_parse_optional_int(form.get("max_new_tokens")) or 512,  # type: ignore[arg-type]
        temperature=_parse_optional_float(form.get("temperature")),  # type: ignore[arg-type]
        top_p=_parse_optional_float(form.get("top_p")),  # type: ignore[arg-type]
        top_k=_parse_optional_int(form.get("top_k")),  # type: ignore[arg-type]
        do_sample=do_sample,
        include_raw=bool(include_raw) if include_raw is not None else False,
        image_max_soft_tokens=image_max_soft_tokens,
    )
    return req, tmp_path, None


async def _video_request_from_multipart(request: Request) -> tuple[VideoRequest, Path | None]:
    form = await request.form()

    text = (form.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Missing form field: text")

    video_url: str | None = None
    chosen_upload: Any | None = None

    video_url_field = form.get("video_url")
    if _is_starlette_upload_file(video_url_field):
        chosen_upload = video_url_field
    elif video_url_field is None:
        video_url = None
    else:
        video_url = str(video_url_field).strip() or None
        if video_url and video_url.startswith("UploadFile("):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Malformed multipart request: `video_url` was not parsed as a file upload. "
                    "Use: -F \"video_url=@your.mp4\" (note the '@'), or use -F \"video_file=@your.mp4\"."
                ),
            )

    video_file_field = form.get("video_file")
    if chosen_upload is None and _is_starlette_upload_file(video_file_field):
        chosen_upload = video_file_field
    elif chosen_upload is None and video_file_field is not None and not _is_starlette_upload_file(video_file_field):
        # Allow path-like strings under rare clients; most users should upload bytes with @.
        raise HTTPException(
            status_code=400,
            detail="`video_file` must be a file upload (use -F \"video_file=@movie.mp4\").",
        )

    uploads: list[Any] = []
    for _, v in form.multi_items():
        if _is_starlette_upload_file(v):
            uploads.append(v)

    if chosen_upload is None:
        if len(uploads) == 1:
            chosen_upload = uploads[0]
        elif len(uploads) > 1:
            raise HTTPException(
                status_code=400,
                detail="Multiple file parts were uploaded; specify `video_url=@...` or `video_file=@...`.",
            )

    tmp_path: Path | None = None
    uploaded_abs_path: str | None = None
    if chosen_upload is not None:
        if video_url:
            raise HTTPException(status_code=400, detail="Provide only one of: video_url (string) or a video file upload.")

        up = chosen_upload
        raw_name = (up.filename or "upload").strip() or "upload"
        safe_name = Path(raw_name).name
        suffix = Path(safe_name).suffix.lower()
        allowed = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}
        if suffix and suffix not in allowed:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported uploaded video extension {suffix!r}. Allowed: {sorted(allowed)}",
            )

        data = await up.read()
        if not data:
            raise HTTPException(status_code=400, detail="Uploaded video file is empty")

        uploads_dir = _default_upload_tmp_dir()
        tmp_path = uploads_dir / f"{uuid.uuid4().hex}{suffix or '.mp4'}"
        tmp_path.write_bytes(data)
        uploaded_abs_path = str(tmp_path.resolve())

    enable_thinking = _truthy_form_value(form.get("enable_thinking"))  # type: ignore[arg-type]
    do_sample = _truthy_form_value(form.get("do_sample"))  # type: ignore[arg-type]
    include_raw = _truthy_form_value(form.get("include_raw"))  # type: ignore[arg-type]
    video_num_frames = _parse_optional_int(form.get("video_num_frames"))  # type: ignore[arg-type]
    video_max_soft_tokens = _parse_optional_int(form.get("video_max_soft_tokens"))  # type: ignore[arg-type]

    if uploaded_abs_path is not None:
        req = VideoRequest(
            video_url=None,
            video_upload_path=uploaded_abs_path,
            text=text,
            enable_thinking=bool(enable_thinking) if enable_thinking is not None else False,
            max_new_tokens=_parse_optional_int(form.get("max_new_tokens")) or 512,  # type: ignore[arg-type]
            temperature=_parse_optional_float(form.get("temperature")),  # type: ignore[arg-type]
            top_p=_parse_optional_float(form.get("top_p")),  # type: ignore[arg-type]
            top_k=_parse_optional_int(form.get("top_k")),  # type: ignore[arg-type]
            do_sample=do_sample,
            include_raw=bool(include_raw) if include_raw is not None else False,
            video_num_frames=video_num_frames,
            video_max_soft_tokens=video_max_soft_tokens,
        )
        return req, tmp_path

    if not video_url:
        raise HTTPException(
            status_code=400,
            detail="Missing video input: provide `video_url` as http(s) string, or upload a file via `video_url=@...`.",
        )

    req = VideoRequest(
        video_url=video_url,
        video_upload_path=None,
        text=text,
        enable_thinking=bool(enable_thinking) if enable_thinking is not None else False,
        max_new_tokens=_parse_optional_int(form.get("max_new_tokens")) or 512,  # type: ignore[arg-type]
        temperature=_parse_optional_float(form.get("temperature")),  # type: ignore[arg-type]
        top_p=_parse_optional_float(form.get("top_p")),  # type: ignore[arg-type]
        top_k=_parse_optional_int(form.get("top_k")),  # type: ignore[arg-type]
        do_sample=do_sample,
        include_raw=bool(include_raw) if include_raw is not None else False,
        video_num_frames=video_num_frames,
        video_max_soft_tokens=video_max_soft_tokens,
    )
    return req, None


async def _chat_request_from_multipart(request: Request) -> ChatRequest:
    form = await request.form()

    # Reject accidental file parts early (multipart can include uploads; chat expects fields only).
    for _, v in form.multi_items():
        if _is_starlette_upload_file(v):
            raise HTTPException(
                status_code=400,
                detail="Multipart /chat/stream does not accept file uploads. Put JSON in `messages` or use field `text`.",
            )

    messages_raw = form.get("messages")
    text = (form.get("text") or "").strip()

    messages: list[ChatMessage]
    if messages_raw is not None:
        if text:
            raise HTTPException(status_code=400, detail="Provide only one of: `messages` (JSON string) or `text`.")
        if not str(messages_raw).strip():
            raise HTTPException(status_code=400, detail="`messages` is empty")
        try:
            data = json.loads(str(messages_raw))
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"`messages` must be valid JSON: {e}")
        if not isinstance(data, list) or not data:
            raise HTTPException(status_code=400, detail="`messages` must be a non-empty JSON array")
        try:
            messages = [ChatMessage.model_validate(m) for m in data]
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=_json_safe_validation_detail(e.errors()))
    else:
        if not text:
            raise HTTPException(status_code=400, detail="Missing form field: text (or provide `messages` as JSON)")
        messages = [ChatMessage(role="user", content=text)]

    enable_thinking = _truthy_form_value(form.get("enable_thinking"))  # type: ignore[arg-type]
    do_sample = _truthy_form_value(form.get("do_sample"))  # type: ignore[arg-type]
    include_raw = _truthy_form_value(form.get("include_raw"))  # type: ignore[arg-type]

    return ChatRequest(
        messages=messages,
        enable_thinking=bool(enable_thinking) if enable_thinking is not None else False,
        max_new_tokens=_parse_optional_int(form.get("max_new_tokens")) or 1024,  # type: ignore[arg-type]
        temperature=_parse_optional_float(form.get("temperature")),  # type: ignore[arg-type]
        top_p=_parse_optional_float(form.get("top_p")),  # type: ignore[arg-type]
        top_k=_parse_optional_int(form.get("top_k")),  # type: ignore[arg-type]
        do_sample=do_sample,
        include_raw=bool(include_raw) if include_raw is not None else False,
    )


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
    global _model, _processor, _model_path, _load_error, _model_kind, _model_supports_audio
    _model_kind = None
    _model_supports_audio = None
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

    _apply_transformers_runtime_patches()
    _configure_torch_runtime_for_inference()

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
            log.info(
                "Model load: max_memory=%r (larger GEMMA_MAX_MEMORY_GPU per GPU => fewer layers on CPU if they fit)",
                mm,
            )

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
    try:
        model.eval()
    except Exception:
        pass
    _model = model
    _model_path = path
    _load_error = None
    _model_supports_audio = getattr(_model.config, "audio_config", None) is not None
    log.info(
        "Model load: audio_config %s",
        "present" if _model_supports_audio else "absent (null)",
    )
    _startup_echo(
        f"stage=model_load_complete total_s={time.perf_counter() - t0:.1f} model_kind={_model_kind!r} "
        f"audio={_model_supports_audio!r} — HTTP API ready"
    )
    log.info(
        "Model load: complete in %.1fs — ready for stream routes (%s)",
        time.perf_counter() - t0,
        _model_kind,
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
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
    if _api_backend() == "vllm":
        try:
            import gemma_vllm as gv

            gv.load_vllm_backend()
            if gv.load_error():
                _startup_echo(f"stage=vllm_load_failed error={gv.load_error()!r}")
                log.warning("vLLM backend failed to load: %s", gv.load_error())
            else:
                _startup_echo("stage=vllm_load_ok")
        except Exception as e:
            _startup_echo(f"stage=vllm_load_exception error={e!r}")
            log.exception("vLLM backend load raised")

    if not _truthy_env("GEMMA_SKIP_HF_MODEL", False):
        _load_model()
    if _load_error:
        _startup_echo(f"stage=lifespan_model_failed error={_load_error!r}")
        log.warning("Startup finished with model error (service degraded): %s", _load_error)
    else:
        if not _truthy_env("GEMMA_SKIP_HF_MODEL", False):
            _startup_echo("stage=lifespan_model_ok — uvicorn will print 'Application startup complete'")
        elif _vllm_ready():
            _startup_echo("stage=hf_model_skipped_vllm_ok — uvicorn will print 'Application startup complete'")
        else:
            _startup_echo("stage=hf_model_skipped")
        log.info("Startup: application ready (uvicorn will log 'Application startup complete')")
    yield
    try:
        if _api_backend() == "vllm":
            import gemma_vllm as gv

            gv.shutdown_vllm()
    except Exception:
        pass


app = FastAPI(title="Gemma 4 local chat API", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def _request_validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    # Default handler tries to json-encode pydantic errors verbatim; `input` may contain raw multipart bytes.
    return JSONResponse(
        status_code=422,
        content={"detail": _json_safe_validation_detail(exc.errors())},
    )


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


# Gemma 4 image/video processors only allow these soft-token budgets (see transformers Gemma4*Processor).
_GEMMA4_VISION_SOFT_TOKEN_CHOICES: frozenset[int] = frozenset({70, 140, 280, 560, 1120})


def _optional_int_env(name: str, *, min_v: int, max_v: int) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        v = int(raw)
    except ValueError:
        log.warning("Invalid integer for %s=%r — ignoring", name, raw)
        return None
    if v < min_v or v > max_v:
        log.warning("%s=%s out of range [%s,%s] — ignoring", name, v, min_v, max_v)
        return None
    return v


def _processor_kwargs_for_image(req: "ImageRequest") -> dict[str, Any]:
    """Extra kwargs for `apply_chat_template(..., processor_kwargs=...)`."""
    images_kw: dict[str, Any] = {}
    if req.image_max_soft_tokens is not None:
        images_kw["max_soft_tokens"] = req.image_max_soft_tokens
    else:
        env_v = _optional_int_env("GEMMA_IMAGE_MAX_SOFT_TOKENS", min_v=70, max_v=1120)
        if env_v is not None and env_v in _GEMMA4_VISION_SOFT_TOKEN_CHOICES:
            images_kw["max_soft_tokens"] = env_v
        elif env_v is not None:
            log.warning(
                "GEMMA_IMAGE_MAX_SOFT_TOKENS=%s is not in %s — ignoring",
                env_v,
                sorted(_GEMMA4_VISION_SOFT_TOKEN_CHOICES),
            )
    if not images_kw:
        return {}
    return {"images_kwargs": images_kw}


def _processor_kwargs_for_video(req: "VideoRequest") -> dict[str, Any]:
    videos_kw: dict[str, Any] = {}
    if req.video_num_frames is not None:
        videos_kw["num_frames"] = req.video_num_frames
    else:
        env_nf = _optional_int_env("GEMMA_VIDEO_NUM_FRAMES", min_v=1, max_v=256)
        if env_nf is not None:
            videos_kw["num_frames"] = env_nf

    if req.video_max_soft_tokens is not None:
        videos_kw["max_soft_tokens"] = req.video_max_soft_tokens
    else:
        env_v = _optional_int_env("GEMMA_VIDEO_MAX_SOFT_TOKENS", min_v=70, max_v=1120)
        if env_v is not None and env_v in _GEMMA4_VISION_SOFT_TOKEN_CHOICES:
            videos_kw["max_soft_tokens"] = env_v
        elif env_v is not None:
            log.warning(
                "GEMMA_VIDEO_MAX_SOFT_TOKENS=%s is not in %s — ignoring",
                env_v,
                sorted(_GEMMA4_VISION_SOFT_TOKEN_CHOICES),
            )

    if not videos_kw:
        return {}
    return {"videos_kwargs": videos_kw}


class ImageRequest(BaseModel):
    image_url: str | None = Field(
        default=None,
        description="HTTP(S) URL of the image (use this or image_file, not both)",
    )
    image_file: str | None = Field(
        default=None,
        description=(
            "Path to an image under the configured images directory (`GEMMA_IMAGES_DIR`). "
            "May be relative (e.g. `photo.png`) or absolute as long as it resolves under that root "
            "(in Docker this is commonly `/images/...`)."
        ),
    )
    image_upload_path: str | None = Field(
        default=None,
        description=(
            "Internal: absolute filesystem path to an uploaded image staged outside `GEMMA_IMAGES_DIR`. "
            "Not intended for JSON clients."
        ),
    )
    text: str = Field(..., description="Question or instruction about the image")
    enable_thinking: bool = False
    max_new_tokens: int = Field(512, ge=1, le=8192)
    temperature: float | None = Field(None, ge=0, le=2)
    top_p: float | None = Field(None, ge=0, le=1)
    top_k: int | None = Field(None, ge=1)
    do_sample: bool | None = None
    include_raw: bool = False
    image_max_soft_tokens: int | None = Field(
        default=None,
        description=(
            "Optional vision token budget per image: one of 70, 140, 280, 560, 1120. "
            "Lower is faster with less spatial detail. Overrides GEMMA_IMAGE_MAX_SOFT_TOKENS."
        ),
    )

    @field_validator("image_max_soft_tokens")
    @classmethod
    def _validate_image_soft_tokens(cls, v: int | None) -> int | None:
        if v is None or v in _GEMMA4_VISION_SOFT_TOKEN_CHOICES:
            return v
        raise ValueError(f"image_max_soft_tokens must be one of {sorted(_GEMMA4_VISION_SOFT_TOKEN_CHOICES)}")

    @model_validator(mode="after")
    def _validate_image_sources(self) -> ImageRequest:
        u = bool((self.image_url or "").strip())
        f = bool((self.image_file or "").strip())
        p = bool((self.image_upload_path or "").strip())
        if int(u) + int(f) + int(p) != 1:
            raise ValueError(
                "Provide exactly one of: image_url, image_file (under GEMMA_IMAGES_DIR), or image_upload_path (internal)."
            )
        return self


class VideoRequest(BaseModel):
    video_url: str | None = Field(
        default=None,
        description="HTTP(S) URL of the video (mp4/webm), OR use multipart upload fields instead.",
    )
    video_upload_path: str | None = Field(
        default=None,
        description="Internal: absolute filesystem path to an uploaded video staged under GEMMA_UPLOAD_TMP_DIR.",
    )
    text: str = Field(..., description="Question or instruction about the video")
    enable_thinking: bool = False
    max_new_tokens: int = Field(512, ge=1, le=8192)
    temperature: float | None = Field(None, ge=0, le=2)
    top_p: float | None = Field(None, ge=0, le=1)
    top_k: int | None = Field(None, ge=1)
    do_sample: bool | None = None
    include_raw: bool = False
    video_num_frames: int | None = Field(
        default=None,
        ge=1,
        le=256,
        description=(
            "Optional: number of frames sampled from the video. Lower is faster (less decode + vision work). "
            "Overrides GEMMA_VIDEO_NUM_FRAMES."
        ),
    )
    video_max_soft_tokens: int | None = Field(
        default=None,
        description=(
            "Optional vision token budget per frame: one of 70, 140, 280, 560, 1120. "
            "Lower is faster. Overrides GEMMA_VIDEO_MAX_SOFT_TOKENS."
        ),
    )

    @field_validator("video_max_soft_tokens")
    @classmethod
    def _validate_video_soft_tokens(cls, v: int | None) -> int | None:
        if v is None or v in _GEMMA4_VISION_SOFT_TOKEN_CHOICES:
            return v
        raise ValueError(f"video_max_soft_tokens must be one of {sorted(_GEMMA4_VISION_SOFT_TOKEN_CHOICES)}")

    @model_validator(mode="after")
    def _validate_video_sources(self) -> VideoRequest:
        u = bool((self.video_url or "").strip())
        p = bool((self.video_upload_path or "").strip())
        if int(u) + int(p) != 1:
            raise ValueError("Provide exactly one of: video_url (remote) or a multipart file upload.")
        return self


def _inference_device() -> Any:
    assert _model is not None
    dev = getattr(_model, "device", None)
    if dev is not None:
        return dev
    for p in _model.parameters():
        if p.device.type == "cuda":
            return p.device
    return next(_model.parameters()).device


def _configure_torch_runtime_for_inference() -> None:
    """
    Enable faster CUDA kernels when available.

    Safe to call multiple times; no-op on CPU-only.
    """
    try:
        import torch
    except Exception:
        return

    if not torch.cuda.is_available():
        return

    # TF32 is a big win on modern NVIDIA GPUs for matmul-heavy models.
    if _truthy_env("GEMMA_ENABLE_TF32", True):
        try:
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
            torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # Prefer fused scaled-dot-product attention implementations when possible.
    # (PyTorch will fall back automatically if unsupported.)
    if _truthy_env("GEMMA_ENABLE_SDPA", True):
        try:
            torch.backends.cuda.enable_flash_sdp(True)  # type: ignore[attr-defined]
            torch.backends.cuda.enable_mem_efficient_sdp(True)  # type: ignore[attr-defined]
            # Keep math SDPA enabled as a compatibility fallback. Some models / shapes
            # (e.g. GQA or large head_dim) cannot use fused kernels and require math SDPA.
            if _truthy_env("GEMMA_DISABLE_MATH_SDPA", False):
                torch.backends.cuda.enable_math_sdp(False)  # type: ignore[attr-defined]
        except Exception:
            pass


def _inference_context():
    """
    Inference-only context shared by all endpoints.

    - torch.inference_mode(): disables autograd overhead
    - autocast(cuda): bf16/fp16 kernels for faster decode
    """
    try:
        import torch
    except Exception:
        return nullcontext()

    stack = ExitStack()
    stack.enter_context(torch.inference_mode())

    if torch.cuda.is_available() and _truthy_env("GEMMA_USE_AUTOCAST", True):
        dtype_raw = os.environ.get("GEMMA_AUTOCAST_DTYPE", "bf16").strip().lower()
        if dtype_raw in ("fp16", "float16", "half"):
            dtype = torch.float16
        else:
            dtype = torch.bfloat16
        stack.enter_context(torch.autocast(device_type="cuda", dtype=dtype))
    return stack


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


def _generate_stream_sse_from_inputs(
    *,
    inputs: Any,
    gen_kwargs: dict[str, Any],
    include_raw: bool,
    meta_extra: dict[str, Any] | None = None,
):
    """
    Shared SSE streamer for any already-prepared `inputs` compatible with `_model.generate(**inputs, ...)`.
    """
    assert _model is not None and _processor is not None

    tokenizer = getattr(_processor, "tokenizer", None)
    if tokenizer is None:
        raise HTTPException(status_code=500, detail="Processor has no tokenizer; streaming not available for this model.")

    try:
        from transformers import TextIteratorStreamer
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming requires transformers TextIteratorStreamer: {e}")

    class _CountingTextIteratorStreamer(TextIteratorStreamer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.prompt_len: int | None = None
            self.new_token_count: int = 0

        def put(self, value):  # type: ignore[override]
            try:
                n = int(getattr(value, "shape", [0])[-1]) if value is not None else 0
            except Exception:
                n = 0
            if self.prompt_len is None:
                self.prompt_len = n
            else:
                self.new_token_count += max(0, n)
            return super().put(value)

    streamer = _CountingTextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=False,
    )

    merged = dict(gen_kwargs)
    merged["streamer"] = streamer

    t0 = time.perf_counter()
    t_started: float | None = None

    def _run_generate() -> None:
        assert _model is not None
        nonlocal t_started
        t_started = time.perf_counter()
        with _gen_lock:
            with _inference_context():
                _model.generate(**inputs, **merged)

    th = threading.Thread(target=_run_generate, name="generate-stream", daemon=True)
    th.start()
    _perf(
        "stream_generate_thread_started delay_ms=%.2f max_new_tokens=%s do_sample=%s",
        (time.perf_counter() - t0) * 1000.0,
        merged.get("max_new_tokens"),
        merged.get("do_sample"),
    )

    meta: dict[str, Any] = {"model_path": str(_model_path) if _model_path else None, "backend": "hf"}
    if meta_extra:
        meta.update(meta_extra)
    yield "event: meta\ndata: " + json.dumps(meta) + "\n\n"

    raw_parts: list[str] = []
    ttft: float | None = None
    n_chars = 0

    try:
        for chunk in streamer:
            if ttft is None:
                ttft = round(time.perf_counter() - t0, 4)
                _perf(
                    "stream_first_chunk ttft_s=%s thread_start_delay_ms=%.2f",
                    ttft,
                    ((t_started - t0) * 1000.0) if t_started is not None else -1.0,
                )
            if not chunk:
                continue
            raw_parts.append(chunk)
            n_chars += len(chunk)
            yield "event: token\ndata: " + json.dumps({"text": chunk}) + "\n\n"
    except Exception as e:
        yield "event: error\ndata: " + json.dumps({"detail": f"{e.__class__.__name__}: {e}"}) + "\n\n"
        return
    finally:
        th.join(timeout=0.1)

    raw = "".join(raw_parts)
    parsed = _processor.parse_response(raw)

    elapsed = max(1e-9, time.perf_counter() - t0)
    cps = round(n_chars / elapsed, 3)
    tps = round(float(getattr(streamer, "new_token_count", 0)) / elapsed, 3)

    yield (
        "event: done\ndata: "
        + json.dumps(
            {
                "raw": raw if include_raw else None,
                "parsed": parsed,
                "model_path": str(_model_path) if _model_path else None,
                "time_to_first_token_seconds": ttft,
                "tokens_per_second": tps,
                "chars_per_second": cps,
            }
        )
        + "\n\n"
    )


def _chat_stream_sse_events(req: ChatRequest):
    """
    Yield Server-Sent Events (SSE) for chat completion.

    Event payloads are JSON strings:
      - event: meta   data: {"model_path": "..."}
      - event: token  data: {"text": "..."}
      - event: done   data: {"raw": "...", "parsed": ..., "time_to_first_token_seconds": ..., "tokens_per_second": ...}
      - event: error  data: {"detail": "..."}
    """
    assert _model is not None and _processor is not None

    t0 = time.perf_counter()
    raw_messages = [m.model_dump() for m in req.messages]
    default_system_applied = default_system_prompt_would_prepend(raw_messages)
    messages = prepend_default_system(raw_messages, multimodal=False)
    prompt = _processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=req.enable_thinking,
    )
    t_prompt = time.perf_counter()
    inputs = _processor(text=prompt, return_tensors="pt")
    t_tok = time.perf_counter()
    inputs = inputs.to(_inference_device())
    t_to = time.perf_counter()
    try:
        input_len = int(inputs["input_ids"].shape[-1])
    except Exception:
        input_len = -1
    _perf(
        "chat_stream_prepare prompt_ms=%.2f tokenize_ms=%.2f to_device_ms=%.2f input_len=%s",
        (t_prompt - t0) * 1000.0,
        (t_tok - t_prompt) * 1000.0,
        (t_to - t_tok) * 1000.0,
        input_len,
    )

    gen_kwargs = _build_generation_kwargs(
        req.max_new_tokens,
        req.temperature,
        req.top_p,
        req.top_k,
        req.do_sample,
    )
    yield from _generate_stream_sse_from_inputs(
        inputs=inputs,
        gen_kwargs=gen_kwargs,
        include_raw=req.include_raw,
        meta_extra={"default_system_prompt_applied": default_system_applied},
    )


def _image_content_block(req: ImageRequest) -> dict[str, str]:
    u = (req.image_url or "").strip()
    f = (req.image_file or "").strip()
    p = (req.image_upload_path or "").strip()

    # `ImageRequest` already enforces exactly-one via model_validator, but keep server-side checks explicit.
    if int(bool(u)) + int(bool(f)) + int(bool(p)) != 1:
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of: image_url, image_file (under GEMMA_IMAGES_DIR), or multipart upload.",
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

    if p:
        path = Path(p).expanduser().resolve()
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"Uploaded image not found: {path}")
        return {"type": "image", "path": str(path)}

    path = _resolve_image_under_images_dir(f)
    return {"type": "image", "path": str(path)}


@app.get("/health")
def health():
    vllm_ok = _vllm_ready()
    ok = (_model is not None and _processor is not None) or vllm_ok
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
    vllm_detail: dict[str, Any] = {"backend": _api_backend()}
    try:
        import gemma_vllm as gv

        vllm_detail["vllm_loaded"] = gv.is_loaded()
        if gv.load_error():
            vllm_detail["vllm_error"] = gv.load_error()
        if gv.model_path_str():
            vllm_detail["vllm_model_path"] = gv.model_path_str()
    except Exception as e:
        vllm_detail["vllm_import_error"] = str(e)

    return {
        "status": "ok" if ok else "degraded",
        "model_loaded": ok,
        "model_kind": _model_kind,
        "image_endpoint": ok and (_model_kind == "multimodal" or vllm_ok),
        "audio_supported": bool(ok and _model_supports_audio),
        "images_dir": str(_default_images_dir()),
        "gpu": gpu_block,
        "error": _load_error,
        "model_path": str(_model_path) if _model_path else None,
        "vllm": vllm_detail,
        "default_system_prompt_configured": bool(default_system_prompt_text()),
        "offload_hints": {
            "GEMMA_DEVICE_MAP": os.environ.get("GEMMA_DEVICE_MAP", "auto"),
            "GEMMA_MAX_MEMORY_GPU": os.environ.get("GEMMA_MAX_MEMORY_GPU", ""),
            "GEMMA_MAX_MEMORY_CPU": os.environ.get("GEMMA_MAX_MEMORY_CPU", ""),
            "GEMMA_LOAD_4BIT": _truthy_env("GEMMA_LOAD_4BIT", False),
            "GEMMA_LOAD_8BIT": _truthy_env("GEMMA_LOAD_8BIT", False),
            "GEMMA_VIDEO_NUM_FRAMES": os.environ.get("GEMMA_VIDEO_NUM_FRAMES", ""),
            "GEMMA_VIDEO_MAX_SOFT_TOKENS": os.environ.get("GEMMA_VIDEO_MAX_SOFT_TOKENS", ""),
            "GEMMA_IMAGE_MAX_SOFT_TOKENS": os.environ.get("GEMMA_IMAGE_MAX_SOFT_TOKENS", ""),
        },
    }


@app.post("/chat/stream")
async def chat_stream(
    request: Request,
    _: None = Depends(require_api_key),
):
    ct = (request.headers.get("content-type") or "").lower()
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty request body")

    is_multipart = ("multipart/form-data" in ct) or ("boundary=" in ct) or _body_looks_like_multipart(body)
    if is_multipart:
        req = await _chat_request_from_multipart(request)
    else:
        req = _parse_json_model(ChatRequest, body, route="POST /chat/stream")  # type: ignore[assignment]

    if _vllm_ready():
        import gemma_vllm as gv

        async def _vllm_chat_sse():
            async for chunk in gv.sse_chat_stream(
                messages=[m.model_dump() for m in req.messages],
                enable_thinking=req.enable_thinking,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                do_sample=req.do_sample,
                include_raw=req.include_raw,
            ):
                yield chunk

        return StreamingResponse(
            _vllm_chat_sse(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    if _model is None or _processor is None:
        raise HTTPException(status_code=503, detail=_load_error or "Model not loaded")

    return StreamingResponse(
        _chat_stream_sse_events(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Helps reverse proxies (nginx) not buffer; harmless otherwise.
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/image/stream")
async def image_chat_stream(
    request: Request,
    _: None = Depends(require_api_key),
):
    ct = (request.headers.get("content-type") or "").lower()
    tmp_path: Path | None = None
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty request body")

    is_multipart = ("multipart/form-data" in ct) or ("boundary=" in ct) or _body_looks_like_multipart(body)
    if is_multipart:
        req, tmp_path, _ = await _image_request_from_multipart(request)
    else:
        req = _parse_json_model(ImageRequest, body, route="POST /image/stream")  # type: ignore[assignment]

    assert req is not None
    image_block = _image_content_block(req)
    if _vllm_ready():
        import gemma_vllm as gv

        async def _vllm_image_sse():
            try:
                async for chunk in gv.sse_image_stream(
                    image_block=image_block,
                    text=req.text,
                    enable_thinking=req.enable_thinking,
                    image_max_soft_tokens=req.image_max_soft_tokens,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                    do_sample=req.do_sample,
                    include_raw=req.include_raw,
                ):
                    yield chunk
            finally:
                if tmp_path is not None:
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except Exception:
                        log.warning("Image(stream): could not delete temp upload: %s", tmp_path)

        return StreamingResponse(
            _vllm_image_sse(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
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

        t0 = time.perf_counter()
        base_messages = [
            {
                "role": "user",
                "content": [
                    image_block,
                    {"type": "text", "text": req.text},
                ],
            }
        ]
        default_system_applied = default_system_prompt_would_prepend(base_messages)
        messages = prepend_default_system(base_messages, multimodal=True)
        tpl_kw: dict[str, Any] = {
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "add_generation_prompt": True,
        }
        mm_pk = _processor_kwargs_for_image(req)
        try:
            if mm_pk:
                inputs = _processor.apply_chat_template(
                    messages, **tpl_kw, enable_thinking=req.enable_thinking, processor_kwargs=mm_pk
                )
            else:
                inputs = _processor.apply_chat_template(messages, **tpl_kw, enable_thinking=req.enable_thinking)
        except TypeError:
            if mm_pk:
                inputs = _processor.apply_chat_template(messages, **tpl_kw, processor_kwargs=mm_pk)
            else:
                inputs = _processor.apply_chat_template(messages, **tpl_kw)

        t_tpl = time.perf_counter()
        inputs = inputs.to(_inference_device())
        t_to = time.perf_counter()
        try:
            input_len = int(inputs["input_ids"].shape[-1])
        except Exception:
            input_len = -1
        _perf(
            "image_stream_prepare template_ms=%.2f to_device_ms=%.2f input_len=%s",
            (t_tpl - t0) * 1000.0,
            (t_to - t_tpl) * 1000.0,
            input_len,
        )
        gen_kwargs = _build_generation_kwargs(
            req.max_new_tokens,
            req.temperature,
            req.top_p,
            req.top_k,
            req.do_sample,
        )

        return StreamingResponse(
            _generate_stream_sse_from_inputs(
                inputs=inputs,
                gen_kwargs=gen_kwargs,
                include_raw=req.include_raw,
                meta_extra={"default_system_prompt_applied": default_system_applied},
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                log.warning("Image(stream): could not delete temp upload: %s", tmp_path)


@app.post("/video/stream")
async def video_chat_stream(
    request: Request,
    _: None = Depends(require_api_key),
):
    ct = (request.headers.get("content-type") or "").lower()
    tmp_path: Path | None = None
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty request body")

    is_multipart = ("multipart/form-data" in ct) or ("boundary=" in ct) or _body_looks_like_multipart(body)
    if is_multipart:
        req, tmp_path = await _video_request_from_multipart(request)
    else:
        req = _parse_json_model(VideoRequest, body, route="POST /video/stream")  # type: ignore[assignment]

    upload_path = (req.video_upload_path or "").strip()
    video_url = (req.video_url or "").strip()
    if upload_path:
        p = Path(upload_path).expanduser().resolve()
        if not p.is_file():
            raise HTTPException(status_code=404, detail=f"Uploaded video not found: {p}")
        video_ref = str(p)
    else:
        if not video_url:
            raise HTTPException(status_code=400, detail="video_url is required")
        _validate_remote_media_url(video_url, kind="video")
        video_ref = video_url

    if _vllm_ready():
        import gemma_vllm as gv

        async def _vllm_video_sse():
            try:
                async for chunk in gv.sse_video_stream(
                    video_ref=video_ref,
                    text=req.text,
                    enable_thinking=req.enable_thinking,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                    do_sample=req.do_sample,
                    include_raw=req.include_raw,
                ):
                    yield chunk
            finally:
                if tmp_path is not None:
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except Exception:
                        log.warning("Video(stream): could not delete temp upload: %s", tmp_path)

        return StreamingResponse(
            _vllm_video_sse(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

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

    base_messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_ref},
                {"type": "text", "text": req.text},
            ],
        }
    ]
    default_system_applied = default_system_prompt_would_prepend(base_messages)
    messages = prepend_default_system(base_messages, multimodal=True)
    t0 = time.perf_counter()
    tpl_kw: dict[str, Any] = {
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "add_generation_prompt": True,
    }
    mm_pk = _processor_kwargs_for_video(req)
    try:
        try:
            if mm_pk:
                inputs = _processor.apply_chat_template(
                    messages, **tpl_kw, enable_thinking=req.enable_thinking, processor_kwargs=mm_pk
                )
            else:
                inputs = _processor.apply_chat_template(messages, **tpl_kw, enable_thinking=req.enable_thinking)
        except TypeError:
            if mm_pk:
                inputs = _processor.apply_chat_template(messages, **tpl_kw, processor_kwargs=mm_pk)
            else:
                inputs = _processor.apply_chat_template(messages, **tpl_kw)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Video(stream): processor/apply_chat_template failed")
        raise HTTPException(
            status_code=502,
            detail=(
                "Failed to fetch/process video input inside the server. "
                f"{e.__class__.__name__}: {e}"
            ),
        )

    t_tpl = time.perf_counter()
    inputs = inputs.to(_inference_device())
    t_to = time.perf_counter()
    try:
        input_len = int(inputs["input_ids"].shape[-1])
    except Exception:
        input_len = -1
    _perf(
        "video_stream_prepare template_ms=%.2f to_device_ms=%.2f input_len=%s",
        (t_tpl - t0) * 1000.0,
        (t_to - t_tpl) * 1000.0,
        input_len,
    )
    gen_kwargs = _build_generation_kwargs(
        req.max_new_tokens,
        req.temperature,
        req.top_p,
        req.top_k,
        req.do_sample,
    )

    try:
        return StreamingResponse(
            _generate_stream_sse_from_inputs(
                inputs=inputs,
                gen_kwargs=gen_kwargs,
                include_raw=req.include_raw,
                meta_extra={"default_system_prompt_applied": default_system_applied},
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                log.warning("Video(stream): could not delete temp upload: %s", tmp_path)


@app.get("/")
def root():
    return {
        "service": "gemma-4-local-chat",
        "endpoints": {
            "POST /chat/stream": "text chat — SSE (JSON or multipart: text=... OR messages=<json>)",
            "POST /image/stream": "image + text — SSE (JSON or multipart; multimodal required)",
            "POST /video/stream": "video + text — SSE (JSON or multipart; multimodal required)",
            "GET /health": "load status",
        },
        "model_path_env": "GEMMA_MODEL_PATH (default: ./gemma-4-E4B-it next to api_server.py)",
        "images_dir_env": "GEMMA_IMAGES_DIR (default: ./images next to api_server.py)",
    }
