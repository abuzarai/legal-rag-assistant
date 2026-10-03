"""Rate-limit-aware Gemini client for eval infrastructure.

The API keys available have small per-project rate limits. This module spreads
calls across multiple keys (separate projects) and enforces spacing
client-side, with 429 retryDelay handling and 5xx backoff.

Nothing here touches the production RAG generation path; that runs through the
real `src.backend.rag` code on the production key.
"""

from __future__ import annotations

import re
import threading
import time

from google import genai
from google.genai import types

from evals import config

_RETRY_RE = re.compile(r"retryDelay'?:\s*'?(\d+(?:\.\d+)?)s")


class DailyQuotaExceeded(RuntimeError):
    """The model's daily quota is exhausted; stop instead of retrying."""


class Pool:
    def __init__(self, model: str, keys: list[str], rpm: int = config.INFRA_RPM):
        if not keys:
            raise ValueError("Pool needs at least one key")
        self.model = model
        self.interval = 60.0 / rpm
        self._lock = threading.Lock()
        self._clients: list[genai.Client] = []
        self._next_at: list[float] = []
        for k in keys:
            self._clients.append(genai.Client(api_key=k))
            self._next_at.append(0.0)

    def _reserve(self) -> tuple[genai.Client, int, float]:
        with self._lock:
            now = time.monotonic()
            idx = min(range(len(self._clients)), key=lambda i: self._next_at[i])
            start = max(now, self._next_at[idx])
            self._next_at[idx] = start + self.interval
        return self._clients[idx], idx, max(0.0, start - time.monotonic())

    def _penalize(self, idx: int, delay: float) -> None:
        with self._lock:
            self._next_at[idx] = max(self._next_at[idx], time.monotonic() + delay)

    def _run(self, kind: str, contents, gen_config, max_retries: int):
        last: Exception | None = None
        for attempt in range(max_retries):
            client, idx, wait = self._reserve()
            if wait > 0:
                time.sleep(wait)
            try:
                if kind == "generate":
                    return client.models.generate_content(
                        model=self.model, contents=contents, config=gen_config
                    )
                return client.models.embed_content(
                    model=self.model, contents=contents, config=gen_config
                )
            except Exception as exc:
                msg = str(exc)
                last = exc
                if "404" in msg or "NOT_FOUND" in msg or "400" in msg:
                    raise
                m = _RETRY_RE.search(msg)
                delay = float(m.group(1)) + 1.0 if m else min(2.0**attempt, 45.0)
                if "PerDay" in msg or "per_day" in msg.lower() or delay > 120:
                    raise DailyQuotaExceeded(
                        f"daily quota likely exhausted ({msg[:160]})"
                    ) from exc
                self._penalize(idx, delay)
        assert last is not None
        raise last

    def generate(self, contents, gen_config=None, max_retries: int = 12):
        return self._run("generate", contents, gen_config, max_retries)

    def embed(self, contents, embed_config=None, max_retries: int = 12):
        return self._run("embed", contents, embed_config, max_retries)


_POOLS: dict[tuple[str, str], Pool] = {}
_POOL_LOCK = threading.Lock()


def pool(pool_name: str, model: str) -> Pool:
    """Get a rate-limited pool. `pool_name` maps to one or more key names."""
    keys = config.keys()
    names = {
        "gen": ["drafting", "voice"],
        "judge": ["voice", "drafting"],
        "frontier": ["drafting", "voice"],
        "embed": ["drafting", "voice"],
    }[pool_name]
    selected = [keys[n] for n in names if n in keys]
    ck = (pool_name, model)
    with _POOL_LOCK:
        if ck not in _POOLS:
            _POOLS[ck] = Pool(model, selected)
        return _POOLS[ck]


def generate_json(
    pool_name: str,
    model: str,
    prompt: str,
    schema: dict,
    system: str | None = None,
    temperature: float = 0.0,
) -> tuple[dict, int, int]:
    """Return (parsed_json, input_tokens, output_tokens)."""
    p = pool(pool_name, model)
    cfg = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=schema,
    )
    if system:
        cfg.system_instruction = system
    resp = p.generate(prompt, cfg)
    usage = resp.usage_metadata
    return (
        __import__("json").loads(resp.text),
        getattr(usage, "prompt_token_count", 0) or 0,
        getattr(usage, "response_token_count", 0) or 0,
    )


def generate_text(
    pool_name: str,
    model: str,
    prompt: str,
    system: str | None = None,
    temperature: float = 0.0,
) -> tuple[str, int, int]:
    p = pool(pool_name, model)
    cfg = types.GenerateContentConfig(temperature=temperature)
    if system:
        cfg.system_instruction = system
    resp = p.generate(prompt, cfg)
    usage = resp.usage_metadata
    return (
        resp.text or "",
        getattr(usage, "prompt_token_count", 0) or 0,
        getattr(usage, "response_token_count", 0) or 0,
    )


def embed_batch(
    pool_name: str, model: str, texts: list[str], dims: int, task_type: str
):
    p = pool(pool_name, model)
    result = p.embed(
        [{"parts": [{"text": t}]} for t in texts],
        {"outputDimensionality": dims, "taskType": task_type},
    )
    return [list(e.values) for e in result.embeddings]
