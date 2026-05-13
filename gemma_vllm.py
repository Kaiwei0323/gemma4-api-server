"""
vLLM backend for Gemma 4 (used when GEMMA_API_BACKEND=vllm).

Loads one AsyncLLM (V1 engine) with the same knobs as the reference LLM(...) snippets:
tensor parallel, max_model_len, gpu_memory_utilization, multimodal limits, hf_overrides, etc.

Stream endpoints use token-by-token streaming (RequestOutputKind.DELTA).
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

from gemma_prompts import default_system_prompt_would_prepend, prepend_default_system

log = logging.getLogger(__name__)

_engine: Any = None
_processor: Any = None
_model_path: str | None = None
_load_error: str | None = None


def _truthy(key: str, default: bool = False) -> bool:
    raw = os.environ.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("Invalid float for %s=%r — using %s", name, raw, default)
        return default


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("Invalid int for %s=%r — using %s", name, raw, default)
        return default


def _vllm_device() -> str:
    """
    Choose a non-empty vLLM device string.

    vLLM EngineArgs includes `device` (choices: auto/cuda/cpu/...). In some
    environments vLLM's platform auto-detection can yield an empty device type,
    which later crashes when `torch.device("")` is constructed. Force a sane
    default here, while still allowing an override via env.
    """
    raw = os.environ.get("GEMMA_VLLM_DEVICE", "").strip().lower()
    if raw:
        return raw
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _force_vllm_platform_device_type(device: str) -> None:
    """
    Work around vLLM builds where platform auto-detection yields an empty device
    type (which later crashes as torch.device("")).

    Some vLLM versions do not expose `device` as an EngineArgs/AsyncEngineArgs
    field, so we can't rely on passing `device=...` to AsyncEngineArgs. In those
    cases we patch `current_platform.device_type` if it is empty.
    """
    try:
        from vllm.platforms import current_platform  # type: ignore

        cur = getattr(current_platform, "device_type", None)
        if cur is None or str(cur).strip() == "":
            setattr(current_platform, "device_type", device)
    except Exception:
        # Best-effort only; if this fails and vLLM doesn't accept device kwarg,
        # vLLM may still fail and we'll surface that error.
        return


def _async_engine_args_accepts_device() -> bool:
    try:
        from vllm.engine.arg_utils import AsyncEngineArgs  # type: ignore

        fields = getattr(AsyncEngineArgs, "__dataclass_fields__", None)
        return bool(fields and "device" in fields)
    except Exception:
        return False


def is_loaded() -> bool:
    return _engine is not None and _load_error is None


def load_error() -> str | None:
    return _load_error


def model_path_str() -> str | None:
    return _model_path


def load_vllm_backend() -> None:
    """Initialize AsyncLLM + tokenizer + processor. Sets global _load_error on failure."""
    global _engine, _processor, _model_path, _load_error
    _engine = None
    _processor = None
    _model_path = None
    _load_error = None

    raw = os.environ.get("GEMMA_MODEL_PATH", "").strip()
    from pathlib import Path

    path = Path(raw).expanduser().resolve() if raw else Path(__file__).resolve().parent / "gemma-4-E4B-it"
    if not path.is_dir() or not (path / "config.json").is_file():
        _load_error = f"vLLM: invalid model directory: {path}"
        log.error(_load_error)
        return

    _model_path = str(path)

    try:
        from transformers import AutoProcessor
        from vllm.engine.arg_utils import AsyncEngineArgs
    except Exception as e:
        _load_error = f"vLLM imports failed: {e}"
        log.exception(_load_error)
        return

    try:
        from vllm.v1.engine.async_llm import AsyncLLM
    except Exception as e:
        _load_error = (
            f"vLLM AsyncLLM (v1) import failed: {e}. "
            "Use an image that includes vLLM V1 engine (e.g. vllm/vllm-openai:gemma4-cu130)."
        )
        log.exception(_load_error)
        return

    tp = _int_env("GEMMA_VLLM_TP_SIZE", 2)
    max_len = _int_env("GEMMA_VLLM_MAX_MODEL_LEN", 8192)
    gpu_mem = _float_env("GEMMA_VLLM_GPU_MEMORY_UTILIZATION", 0.90)
    trust = _truthy("GEMMA_VLLM_TRUST_REMOTE_CODE", True)
    quant = os.environ.get("GEMMA_VLLM_QUANTIZATION", "").strip() or None
    device = _vllm_device()

    if os.environ.get("GEMMA_REQUIRE_GPU", "1").strip().lower() not in ("0", "false", "no", "off"):
        if device != "cuda":
            _load_error = (
                "vLLM: GEMMA_REQUIRE_GPU is enabled but vLLM device resolved to "
                f"{device!r}. Ensure the container has GPU access (compose `gpus: all`) "
                "or set GEMMA_VLLM_DEVICE=cuda."
            )
            log.error(_load_error)
            return

    # If vLLM can't be told `device=...`, ensure platform device_type isn't empty.
    _force_vllm_platform_device_type(device)

    # Multimodal defaults aligned with reference snippets
    limit_mm = {"image": _int_env("GEMMA_VLLM_LIMIT_MM_IMAGE", 4), "video": _int_env("GEMMA_VLLM_LIMIT_MM_VIDEO", 1)}
    hf_overrides = {
        "vision_config": {"default_output_length": _int_env("GEMMA_VLLM_VISION_DEFAULT_OUTPUT_LENGTH", 1120)},
        "vision_soft_tokens_per_image": _int_env("GEMMA_VLLM_VISION_SOFT_TOKENS_PER_IMAGE", 1120),
    }
    mm_processor_kwargs = {"max_soft_tokens": _int_env("GEMMA_VLLM_MM_MAX_SOFT_TOKENS_DEFAULT", 280)}

    t0 = time.perf_counter()
    try:
        _processor = AutoProcessor.from_pretrained(_model_path, local_files_only=True)
    except Exception as e:
        _load_error = f"vLLM: failed to load processor: {e}"
        log.exception(_load_error)
        return

    engine_kw: dict[str, Any] = {
        "model": _model_path,
        "tensor_parallel_size": tp,
        "max_model_len": max_len,
        "gpu_memory_utilization": gpu_mem,
        "trust_remote_code": trust,
        "limit_mm_per_prompt": limit_mm,
        "hf_overrides": hf_overrides,
        "mm_processor_kwargs": mm_processor_kwargs,
    }
    if _async_engine_args_accepts_device():
        engine_kw["device"] = device
    if quant:
        engine_kw["quantization"] = quant

    try:
        engine_args = AsyncEngineArgs(**engine_kw)
        _engine = AsyncLLM.from_engine_args(engine_args)
    except TypeError:
        # Older vLLM may not accept some kwargs — drop optional multimodal extras
        engine_kw.pop("hf_overrides", None)
        engine_kw.pop("mm_processor_kwargs", None)
        engine_kw.pop("quantization", None)
        try:
            engine_args = AsyncEngineArgs(**engine_kw)
            _engine = AsyncLLM.from_engine_args(engine_args)
            log.warning("vLLM: AsyncEngineArgs ignored hf_overrides/mm_processor_kwargs (unsupported in this vLLM build)")
        except Exception as e:
            _load_error = f"vLLM: AsyncLLM.from_engine_args failed: {e}"
            log.exception(_load_error)
            return
    except Exception as e:
        _load_error = f"vLLM: AsyncLLM.from_engine_args failed: {e}"
        log.exception(_load_error)
        return

    log.info(
        "vLLM backend ready in %.1fs model=%s tp=%s max_len=%s quantization=%s",
        time.perf_counter() - t0,
        _model_path,
        tp,
        max_len,
        quant or "none",
    )


def shutdown_vllm() -> None:
    global _engine
    if _engine is None:
        return
    try:
        _engine.shutdown()
    except Exception as e:
        log.warning("vLLM shutdown: %s", e)
    _engine = None


def _sampling_params(
    *,
    max_new_tokens: int,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    do_sample: bool | None,
) -> Any:
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    resolved_sample = do_sample
    if resolved_sample is None:
        resolved_sample = temperature is not None and temperature > 0

    sp_kw: dict[str, Any] = {
        "max_tokens": max_new_tokens,
        "output_kind": RequestOutputKind.DELTA,
    }
    if resolved_sample:
        sp_kw["temperature"] = float(temperature if temperature is not None else 1.0)
        if top_p is not None:
            sp_kw["top_p"] = float(top_p)
        if top_k is not None:
            sp_kw["top_k"] = int(top_k)
    else:
        sp_kw["temperature"] = 0.0
    return SamplingParams(**sp_kw)


def _parse_with_processor(raw: str) -> Any:
    assert _processor is not None
    try:
        return _processor.parse_response(raw)
    except Exception:
        return raw


def _count_delta_tokens(completion: Any, fallback_text: str) -> int:
    """
    Best-effort delta token count for a vLLM completion in DELTA mode.

    Prefers `len(token_ids)` when the engine exposes it; falls back to
    counting 1 per non-empty text chunk so per-token latency stays meaningful
    even on builds that do not surface delta token ids.
    """
    tids = getattr(completion, "token_ids", None)
    try:
        if tids is not None:
            return len(tids)
    except Exception:
        pass
    return 1 if fallback_text else 0


def _build_latency_payload(
    *,
    t_request_start: float,
    t_first_token: float | None,
    t_end: float,
    n_tokens: int,
) -> dict[str, Any]:
    """Common latency block emitted in the final SSE `done` event."""
    elapsed = max(1e-9, t_end - t_request_start)
    if t_first_token is not None:
        ttft = round(t_first_token - t_request_start, 4)
        decode_total = max(0.0, t_end - t_first_token)
    else:
        ttft = None
        decode_total = 0.0
    decode_per_token: float | None
    if n_tokens > 1:
        decode_per_token = round(decode_total / float(n_tokens - 1), 6)
    else:
        decode_per_token = None
    tps = round(float(n_tokens) / elapsed, 3) if n_tokens > 0 else None
    return {
        "time_to_first_token_seconds": ttft,
        "prompt_processing_latency_seconds": ttft,
        "decode_latency_seconds": round(decode_total, 4),
        "decode_latency_per_token_seconds": decode_per_token,
        "end_to_end_latency_seconds": round(elapsed, 4),
        "tokens_per_second": tps,
        "output_tokens": n_tokens,
    }


async def sse_chat_stream(
    *,
    messages: list[dict[str, Any]],
    enable_thinking: bool,
    max_new_tokens: int,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    do_sample: bool | None,
    include_raw: bool,
) -> AsyncIterator[str]:
    assert _engine is not None and _processor is not None and _model_path is not None

    raw_messages = list(messages)
    default_system_applied = default_system_prompt_would_prepend(raw_messages)
    messages = prepend_default_system(raw_messages, multimodal=False)

    t0 = time.perf_counter()
    try:
        prompt = _processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        prompt = _processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    _perf_ms("vllm_chat_prompt", t0)

    sp = _sampling_params(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=do_sample,
    )
    request_id = f"chat-{uuid.uuid4().hex}"

    yield (
        "event: meta\ndata: "
        + json.dumps(
            {
                "model_path": _model_path,
                "backend": "vllm",
                "default_system_prompt_applied": default_system_applied,
            }
        )
        + "\n\n"
    )

    raw_parts: list[str] = []
    n_chars = 0
    n_tokens = 0
    t_first_token: float | None = None

    try:
        async for output in _engine.generate(request_id=request_id, prompt=prompt, sampling_params=sp):
            for completion in output.outputs:
                chunk = completion.text or ""
                delta_tokens = _count_delta_tokens(completion, chunk)
                if delta_tokens > 0 and t_first_token is None:
                    t_first_token = time.perf_counter()
                n_tokens += delta_tokens
                if chunk:
                    raw_parts.append(chunk)
                    n_chars += len(chunk)
                    yield "event: token\ndata: " + json.dumps({"text": chunk}) + "\n\n"
            if getattr(output, "finished", False):
                break
    except Exception as e:
        yield "event: error\ndata: " + json.dumps({"detail": f"{e.__class__.__name__}: {e}"}) + "\n\n"
        return

    raw = "".join(raw_parts)
    parsed = _parse_with_processor(raw)
    t_end = time.perf_counter()
    elapsed = max(1e-9, t_end - t0)
    cps = round(n_chars / elapsed, 3)
    latency = _build_latency_payload(
        t_request_start=t0,
        t_first_token=t_first_token,
        t_end=t_end,
        n_tokens=n_tokens,
    )

    yield (
        "event: done\ndata: "
        + json.dumps(
            {
                "raw": raw if include_raw else None,
                "parsed": parsed,
                "model_path": _model_path,
                "chars_per_second": cps,
                **latency,
            }
        )
        + "\n\n"
    )


def _load_pil_image(image_block: dict[str, str]) -> Any:
    from PIL import Image

    u = (image_block.get("url") or "").strip()
    p = (image_block.get("path") or "").strip()
    if p:
        img = Image.open(p).convert("RGB")
        return img
    if u:
        import urllib.request

        with urllib.request.urlopen(u, timeout=120) as resp:
            data = resp.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    raise ValueError("image block has no url or path")


async def sse_image_stream(
    *,
    image_block: dict[str, str],
    text: str,
    enable_thinking: bool,
    image_max_soft_tokens: int | None,
    max_new_tokens: int,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    do_sample: bool | None,
    include_raw: bool,
) -> AsyncIterator[str]:
    assert _engine is not None and _processor is not None and _model_path is not None

    t0 = time.perf_counter()
    try:
        image = _load_pil_image(image_block)
    except (FileNotFoundError, OSError, ValueError) as e:
        yield "event: error\ndata: " + json.dumps({"detail": f"{e.__class__.__name__}: {e}"}) + "\n\n"
        return

    base_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        }
    ]
    default_system_applied = default_system_prompt_would_prepend(base_messages)
    messages = prepend_default_system(base_messages, multimodal=True)
    prompt = _processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    mm_kw = {"max_soft_tokens": image_max_soft_tokens} if image_max_soft_tokens is not None else {}
    req_prompt: Any = {
        "prompt": prompt,
        "multi_modal_data": {"image": image},
    }
    if mm_kw:
        req_prompt["mm_processor_kwargs"] = mm_kw

    _perf_ms("vllm_image_prepare", t0)

    sp = _sampling_params(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=do_sample,
    )
    request_id = f"image-{uuid.uuid4().hex}"

    yield (
        "event: meta\ndata: "
        + json.dumps(
            {
                "model_path": _model_path,
                "backend": "vllm",
                "default_system_prompt_applied": default_system_applied,
            }
        )
        + "\n\n"
    )

    raw_parts: list[str] = []
    n_chars = 0
    n_tokens = 0
    t_first_token: float | None = None

    try:
        async for output in _engine.generate(request_id=request_id, prompt=req_prompt, sampling_params=sp):
            for completion in output.outputs:
                chunk = completion.text or ""
                delta_tokens = _count_delta_tokens(completion, chunk)
                if delta_tokens > 0 and t_first_token is None:
                    t_first_token = time.perf_counter()
                n_tokens += delta_tokens
                if chunk:
                    raw_parts.append(chunk)
                    n_chars += len(chunk)
                    yield "event: token\ndata: " + json.dumps({"text": chunk}) + "\n\n"
            if getattr(output, "finished", False):
                break
    except Exception as e:
        yield "event: error\ndata: " + json.dumps({"detail": f"{e.__class__.__name__}: {e}"}) + "\n\n"
        return

    raw = "".join(raw_parts)
    parsed = _parse_with_processor(raw)
    t_end = time.perf_counter()
    elapsed = max(1e-9, t_end - t0)
    cps = round(n_chars / elapsed, 3)
    latency = _build_latency_payload(
        t_request_start=t0,
        t_first_token=t_first_token,
        t_end=t_end,
        n_tokens=n_tokens,
    )

    yield (
        "event: done\ndata: "
        + json.dumps(
            {
                "raw": raw if include_raw else None,
                "parsed": parsed,
                "model_path": _model_path,
                "chars_per_second": cps,
                **latency,
            }
        )
        + "\n\n"
    )


async def sse_video_stream(
    *,
    video_ref: str,
    text: str,
    enable_thinking: bool,
    max_new_tokens: int,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    do_sample: bool | None,
    include_raw: bool,
) -> AsyncIterator[str]:
    assert _engine is not None and _processor is not None and _model_path is not None

    from vllm.multimodal.utils import fetch_video

    t0 = time.perf_counter()
    try:
        video_url = _video_url_for_vllm_fetch(video_ref)
        video_data = fetch_video(video_url)
    except (FileNotFoundError, ValueError, OSError) as e:
        yield "event: error\ndata: " + json.dumps({"detail": f"{e.__class__.__name__}: {e}"}) + "\n\n"
        return

    base_messages = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": text},
            ],
        }
    ]
    default_system_applied = default_system_prompt_would_prepend(base_messages)
    messages = prepend_default_system(base_messages, multimodal=True)
    prompt = _processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    req_prompt: Any = {"prompt": prompt, "multi_modal_data": {"video": [video_data]}}
    _perf_ms("vllm_video_prepare", t0)

    sp = _sampling_params(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=do_sample,
    )
    request_id = f"video-{uuid.uuid4().hex}"

    yield (
        "event: meta\ndata: "
        + json.dumps(
            {
                "model_path": _model_path,
                "backend": "vllm",
                "default_system_prompt_applied": default_system_applied,
            }
        )
        + "\n\n"
    )

    raw_parts: list[str] = []
    n_chars = 0
    n_tokens = 0
    t_first_token: float | None = None

    try:
        async for output in _engine.generate(request_id=request_id, prompt=req_prompt, sampling_params=sp):
            for completion in output.outputs:
                chunk = completion.text or ""
                delta_tokens = _count_delta_tokens(completion, chunk)
                if delta_tokens > 0 and t_first_token is None:
                    t_first_token = time.perf_counter()
                n_tokens += delta_tokens
                if chunk:
                    raw_parts.append(chunk)
                    n_chars += len(chunk)
                    yield "event: token\ndata: " + json.dumps({"text": chunk}) + "\n\n"
            if getattr(output, "finished", False):
                break
    except Exception as e:
        yield "event: error\ndata: " + json.dumps({"detail": f"{e.__class__.__name__}: {e}"}) + "\n\n"
        return

    raw = "".join(raw_parts)
    parsed = _parse_with_processor(raw)
    t_end = time.perf_counter()
    elapsed = max(1e-9, t_end - t0)
    cps = round(n_chars / elapsed, 3)
    latency = _build_latency_payload(
        t_request_start=t0,
        t_first_token=t_first_token,
        t_end=t_end,
        n_tokens=n_tokens,
    )

    yield (
        "event: done\ndata: "
        + json.dumps(
            {
                "raw": raw if include_raw else None,
                "parsed": parsed,
                "model_path": _model_path,
                "chars_per_second": cps,
                **latency,
            }
        )
        + "\n\n"
    )


def _perf_ms(label: str, t_start: float) -> None:
    if not _truthy("GEMMA_PERF_LOG", False):
        return
    log.info("PERF %s wall_ms=%.2f", label, (time.perf_counter() - t_start) * 1000.0)


def _video_url_for_vllm_fetch(video_ref: str) -> str:
    """
    vLLM `fetch_video` only accepts http(s), data, or file URLs — not bare filesystem paths.
    """
    vr = (video_ref or "").strip()
    low = vr.lower()
    if low.startswith(("http://", "https://", "data:", "file:")):
        return vr
    p = Path(vr).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Video file not found: {video_ref}")
    return p.as_uri()
