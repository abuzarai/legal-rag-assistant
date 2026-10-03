"""Shared config for the RAG evaluation harness.

This module is eval-only. It does not modify production code; it imports the
production modules and points them at the production Gemini credentials so the
numbers describe the deployed system, not a copy.

Credential resolution order:
  1. EVAL_GEMINI_API_KEY (explicit override)
  2. RAG_GEMINI_API_KEY from the sibling webapp .env (the production key)
  3. GEMINI_API_KEY already in the environment
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EVALS = REPO / "evals"
DATA = EVALS / "data"  # large derived caches (gitignored via root data/)
RESULTS = EVALS / "results"
LABELS = EVALS / "labels"
EVAL_SET = EVALS / "eval_set.jsonl"  # committed, versioned

def _webapp_env() -> Path:
    """Path to the sibling webapp .env, overridable via EVAL_WEBAPP_ENV."""
    return Path(os.environ.get("EVAL_WEBAPP_ENV", REPO.parent / "webapp" / ".env"))

# ---- model versions actually used (must match the running config) ----
GENERATOR_MODEL = "gemini-2.5-flash"  # production: GEMINI_MODEL in repo/webapp env
# Substitute generators used because the production model's API quota was
# too constrained to also serve the harness.
BASELINE_MODELS = ["gemini-3.5-flash-lite", "gemini-3-flash-preview"]
FRONTIER_MODEL = "gemini-3-flash-preview"
JUDGE_MODEL = "gemini-3.1-flash-lite"  # independent from both generators
GEN_MODEL = "gemini-3.5-flash-lite"  # eval-set question generation
EMBEDDING_MODEL = (
    "gemini-embedding-001"  # matches EMBEDDING_MODEL in the running container
)
EMBEDDING_DIMS = 768

# Rate limits are per project per model (5 RPM observed). Spread phases
# across non-production keys so evaluation traffic does not consume the
# production key's quota.
KEY_POOLS = {
    "gen": "drafting",
    "judge": "voice",
    "frontier": "drafting",
    "system": "rag",
    "embed": "drafting",
}
INFRA_RPM = 5

# ---- retrieval knobs copied from production ----
HYBRID_ALPHA = 0.5
FETCH_MULTIPLIER = 3  # similarity_search: limit=max(k*3, 10)
MIN_FETCH = 10
RAG_STATUTE_CHUNKS = 3  # env RAG_STATUTE_CHUNKS default
RAG_CASELAW_CHUNKS = 2  # env RAG_CASELAW_CHUNKS default
K_EVAL = 10  # ranking cutoff for the retrieval ablation

# ---- pricing, USD per 1M tokens ----
# Source (retrieved 2026-09-14):
#   https://ai.google.dev/gemini-api/docs/pricing  (Standard paid tier)
#   https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing
# gemini-embedding-001 is a legacy model not listed on the ai.google.dev page;
# the Cloud pricing page lists "Gemini Embedding" online at $0.00015 / 1K tokens.
PRICING = {
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50},
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
    "gemini-embedding-001": {"input": 0.15, "output": 0.0},
}
PRICING_SOURCE = "https://ai.google.dev/gemini-api/docs/pricing (retrieved 2026-09-14)"


def _webapp_keys() -> dict:
    out: dict[str, str] = {}
    path = _webapp_env()
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        for prefix, name in (
            ("RAG_GEMINI_API_KEY=", "rag"),
            ("DRAFTING_GEMINI_API_KEY=", "drafting"),
            ("VOICE_GEMINI_API_KEY=", "voice"),
            ("WEAVIATE_API_KEY=", "weaviate_api_key"),
        ):
            if line.startswith(prefix):
                out[name] = line.split("=", 1)[1].strip().strip('"').strip("'")
    return out


def keys() -> dict:
    """Named API keys (webapp/.env plus explicit EVAL_* env overrides)."""
    out = _webapp_keys()
    overrides = {
        "rag": (
            "EVAL_GEMINI_API_KEY",
            "EVAL_GEMINI_API_KEY_RAG",
            "EVAL_GEMINI_API_KEY_SYSTEM",
        ),
        "drafting": ("EVAL_GEMINI_API_KEY_DRAFTING",),
        "voice": ("EVAL_GEMINI_API_KEY_VOICE",),
    }
    for name, envs in overrides.items():
        for env in envs:
            if os.environ.get(env):
                out[name] = os.environ[env]
    return out


def key_for(pool: str) -> str:
    name = KEY_POOLS[pool]
    k = keys().get(name)
    if not k:
        env_override = os.environ.get(f"EVAL_GEMINI_API_KEY_{pool.upper()}")
        if env_override:
            return env_override
        raise RuntimeError(f"no API key for pool {pool!r} (wanted {name!r})")
    return k


def key_fingerprints() -> dict:
    import hashlib

    return {
        name: hashlib.sha256(k.encode()).hexdigest()[:10] for name, k in keys().items()
    }


def load_repo_env() -> None:
    """Load the repo .env for WEAVIATE_URL etc. without clobbering anything."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    load_dotenv(REPO / ".env", override=False)


def resolve_weaviate_url() -> str:
    """Find the production Weaviate.

    Prefers EVAL_WEAVIATE_URL; then the running container named by
    EVAL_WEAVIATE_CONTAINER (default "weaviate"); then WEAVIATE_URL from the
    environment or .env.
    """
    override = os.environ.get("EVAL_WEAVIATE_URL")
    if override:
        return override
    try:
        import subprocess

        ip = subprocess.check_output(
            [
                "docker",
                "inspect",
                os.environ.get("EVAL_WEAVIATE_CONTAINER", "weaviate"),
                "--format",
                "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
            ],
            text=True,
            timeout=10,
        ).strip()
        if ip:
            return f"http://{ip}:8080"
    except Exception:  # noqa: BLE001,S110  # docker unavailable → fall back silently
        pass
    return os.environ.get("WEAVIATE_URL", "http://localhost:8080")


def configure_weaviate() -> str:
    """Point production modules at the live docker Weaviate with its API key."""
    load_repo_env()
    wk = keys()
    if wk.get("weaviate_api_key"):
        os.environ["WEAVIATE_API_KEY"] = wk["weaviate_api_key"]
    url = resolve_weaviate_url()
    os.environ["WEAVIATE_URL"] = url
    os.environ.setdefault("WEAVIATE_GRPC_PORT", "50051")
    os.environ.setdefault("WEAVIATE_COLLECTION", "LegalChunk")
    return url


def init_credentials(role: str = "embed") -> str:
    """Set GEMINI_API_KEY for production modules (embeddings/run_rag).

    Returns a source label for logging (never the key itself).
    """
    configure_weaviate()
    pool = "system" if role == "system" else "embed"
    os.environ["GEMINI_API_KEY"] = key_for(pool)
    return f"webapp/.env:{KEY_POOLS[pool]}"


def load_corpus() -> list[dict]:
    """Load the cached corpus dump (evals/data/corpus.jsonl)."""
    path = DATA / "corpus.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run `python -m evals.dump_corpus` first"
        )
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_eval_set() -> list[dict]:
    if not EVAL_SET.exists():
        raise FileNotFoundError(f"{EVAL_SET} missing — run the eval-set builder first")
    with EVAL_SET.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def ensure_import_path() -> None:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
