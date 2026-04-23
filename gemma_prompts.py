"""
Shared prompt helpers (used by api_server and gemma_vllm).

`GEMMA_DEFAULT_SYSTEM_PROMPT` — when set, a `system` turn with this text is inserted
as the **first** message on **every** request (each POST to a stream route), so the
model always sees your policy/persona for that turn.

`GEMMA_DEFAULT_SYSTEM_RESPECT_CLIENT_SYSTEM=1` — legacy mode: do **not** prepend if
the client already sent any `role: system` message (client controls system alone).

`GEMMA_DEFAULT_SYSTEM_FORCE_PREPEND=1` — always prepend the server system turn (never
skip). Useful with ``RESPECT_CLIENT_SYSTEM=1`` when you still want the server line
first anyway (``FORCE`` wins over ``RESPECT``).

If the server prepends ``system`` and the client’s first message is also ``system``, the
two turns are merged into one so Gemma’s template (no back-to-back ``system``) stays valid.

Use ``default_system_prompt_would_prepend(messages)`` to mirror prepend logic for SSE meta.
"""

from __future__ import annotations

import os
from typing import Any


def default_system_prompt_text() -> str | None:
    s = os.environ.get("GEMMA_DEFAULT_SYSTEM_PROMPT", "").strip()
    return s or None


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _skip_server_system_prepend(messages: list[dict[str, Any]]) -> bool:
    """True if we should leave ``messages`` unchanged (no server default system turn)."""
    if _truthy_env("GEMMA_DEFAULT_SYSTEM_FORCE_PREPEND"):
        return False
    if _truthy_env("GEMMA_DEFAULT_SYSTEM_RESPECT_CLIENT_SYSTEM"):
        return any(str(m.get("role", "")).lower() == "system" for m in messages)
    return False


def _system_content_as_plain_text(msg: dict[str, Any]) -> str:
    """Flatten ``content`` of a system message to plain text for merging."""
    c = msg.get("content")
    if isinstance(c, str):
        return c.strip()
    if isinstance(c, list):
        parts: list[str] = []
        for block in c:
            if isinstance(block, dict):
                t = str(block.get("type", "")).lower()
                if t == "text":
                    parts.append(str(block.get("text", "")).strip())
                elif t == "input_text":
                    parts.append(str(block.get("text", "")).strip())
        return "\n".join(p for p in parts if p).strip()
    return ""


def _merge_consecutive_system_turns(
    messages: list[dict[str, Any]],
    *,
    multimodal: bool,
) -> list[dict[str, Any]]:
    """
    Gemma chat templates reject ``system`` followed immediately by another ``system``.
    Merge adjacent system turns into one (server text first, then client text).
    """
    if len(messages) < 2:
        return messages
    out = list(messages)
    i = 0
    while i + 1 < len(out):
        r0 = str(out[i].get("role", "")).lower()
        r1 = str(out[i + 1].get("role", "")).lower()
        if r0 == "system" and r1 == "system":
            a = _system_content_as_plain_text(out[i])
            b = _system_content_as_plain_text(out[i + 1])
            merged = f"{a}\n\n{b}".strip() if a and b else (a or b)
            if multimodal:
                out[i] = {"role": "system", "content": [{"type": "text", "text": merged}]}
            else:
                out[i] = {"role": "system", "content": merged}
            out.pop(i + 1)
        else:
            i += 1
    return out


def default_system_prompt_would_prepend(messages: list[dict[str, Any]]) -> bool:
    """True if ``prepend_default_system`` will insert a system turn for this message list."""
    if not default_system_prompt_text():
        return False
    return not _skip_server_system_prepend(messages)


def prepend_default_system(
    messages: list[dict[str, Any]],
    *,
    multimodal: bool = False,
) -> list[dict[str, Any]]:
    """
    If `GEMMA_DEFAULT_SYSTEM_PROMPT` is set, prepend that text as the first `system`
    turn on every call (each HTTP request), unless ``RESPECT_CLIENT_SYSTEM`` applies.
    Adjacent ``system`` messages (server + client) are merged into one.

    - Text chat: `content` is a plain string (matches `ChatMessage.model_dump()`).
    - Image / video: `content` is a list of blocks (MedGemma-style), so
      system uses `[{"type":"text","text": ...}]`.
    """
    text = default_system_prompt_text()
    if not text:
        return list(messages)
    if _skip_server_system_prepend(messages):
        return list(messages)
    if multimodal:
        sys_turn: dict[str, Any] = {"role": "system", "content": [{"type": "text", "text": text}]}
    else:
        sys_turn = {"role": "system", "content": text}
    combined = [sys_turn, *list(messages)]
    return _merge_consecutive_system_turns(combined, multimodal=multimodal)
