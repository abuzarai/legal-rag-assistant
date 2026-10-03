"""Retrieval adapters for the evaluation.

All query embeddings use task_type=RETRIEVAL_QUERY (what production
`embed_query` does) on gemini-embedding-001 at 768 dims — the exact model
recorded in artifacts/ingestion_state.json. Query vectors are computed once
and shared across configs so config differences are pure retrieval differences.

Configs:
  bm25           Weaviate BM25 over `content`, whole corpus, no rerank
  dense          Weaviate near_vector, whole corpus, no rerank
  hybrid         Weaviate hybrid alpha=0.5, whole corpus, no rerank
  hybrid_rerank  hybrid fetch max(k*3,10) then local cosine rerank (production
                 reranker code path), top-k
  production     statute-first merge: RAG_STATUTE_CHUNKS hybrid from
                 cpc-sections, then RAG_CASELAW_CHUNKS from case-laws; the
                 local cosine rerank is opt-in via rerank=True
"""

from __future__ import annotations

import numpy as np

from evals import config

_CLIENT = None
_COLL = None
_EMBEDDER = None


def _deps():
    global _CLIENT, _COLL, _EMBEDDER
    if _COLL is None:
        from src.backend.deps import get_embeddings
        from src.common.config import get_weaviate_collection
        from src.common.weaviate_client import get_weaviate_client

        _CLIENT = get_weaviate_client()
        _COLL = _CLIENT.collections.get(get_weaviate_collection())
        _EMBEDDER = get_embeddings()
    return _COLL, _EMBEDDER


PROPS = ["content", "source", "page", "drive_id"]


def embed_queries(queries: list[str], batch_size: int = 20) -> dict[str, list[float]]:
    """Embed queries with the production embedding model (RETRIEVAL_QUERY).

    Embedding rate limits are shared across the account's keys,
    so batches are small and 429s are retried with the server's retryDelay.
    """
    import re
    import time

    from src.common.config import get_embedding_model, get_embedding_output_dims

    _, embedder = _deps()
    model = get_embedding_model()
    dims = get_embedding_output_dims()
    client = getattr(embedder, "client", None)
    out: dict[str, list[float]] = {}

    if client is None:  # pragma: no cover - fallback to per-query path
        for q in queries:
            out[q] = embedder.embed_query(q)
        return out

    model_name = (
        f"models/{model}" if not str(model).startswith("models/") else str(model)
    )
    for i in range(0, len(queries), batch_size):
        batch = queries[i : i + batch_size]
        last: Exception | None = None
        for attempt in range(10):
            try:
                result = client.models.embed_content(
                    model=model_name,
                    contents=[{"parts": [{"text": q}]} for q in batch],
                    config={
                        "outputDimensionality": dims,
                        "taskType": "RETRIEVAL_QUERY",
                    },
                )
                for q, emb in zip(batch, result.embeddings):
                    out[q] = list(emb.values)
                break
            except Exception as exc:
                last = exc
                msg = str(exc)
                if "404" in msg or "400" in msg:
                    raise
                m = re.search(r"retryDelay'?:\s*'?(\d+(?:\.\d+)?)s", msg)
                delay = float(m.group(1)) + 1.0 if m else min(2.0**attempt, 30.0)
                if delay > 120:
                    raise
                time.sleep(delay)
        else:
            raise last
    return out


def _obj_to_rec(obj) -> dict:
    props = obj.properties or {}
    return {
        "chunk_id": str(obj.uuid),
        "content": props.get("content") or "",
        "source": props.get("source"),
        "page": str(props.get("page")) if props.get("page") is not None else "?",
        "distance": getattr(getattr(obj, "metadata", None), "distance", None),
        "score": getattr(getattr(obj, "metadata", None), "score", None),
        "vector": _extract_vector(obj.vector),
    }


def _extract_vector(vec):
    if isinstance(vec, dict):
        vec = vec.get("default")
    return list(vec) if vec is not None else None


def _cosine(a: list[float], b: list[float]) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def _rerank(vector: list[float], recs: list[dict]) -> list[dict]:
    scored = [(r, _cosine(vector, r["vector"])) for r in recs if r.get("vector")]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [r for r, _ in scored] or recs


def _category_filter(category: str):
    from weaviate.classes.query import Filter

    return Filter.by_property("category").equal(category)


def run_config(
    query: str,
    vector: list[float],
    mode: str,
    k: int = config.K_EVAL,
    k_s: int = config.RAG_STATUTE_CHUNKS,
    k_c: int = config.RAG_CASELAW_CHUNKS,
    rerank: bool = False,
) -> list[dict]:
    """Return a ranked list of chunk records for one query under one config."""
    coll, _ = _deps()

    if mode == "bm25":
        resp = coll.query.bm25(
            query=query,
            limit=k,
            return_properties=PROPS,
            return_metadata=["score"],
        )
        return [_obj_to_rec(o) for o in (resp.objects or [])][:k]

    if mode == "dense":
        resp = coll.query.near_vector(
            near_vector=vector,
            limit=k,
            return_properties=PROPS,
            return_metadata=["distance"],
        )
        return [_obj_to_rec(o) for o in (resp.objects or [])][:k]

    if mode in ("hybrid", "hybrid_rerank"):
        fetch = max(k * config.FETCH_MULTIPLIER, config.MIN_FETCH)
        resp = coll.query.hybrid(
            query=query,
            vector=vector,
            alpha=config.HYBRID_ALPHA,
            limit=fetch,
            return_properties=PROPS,
            return_metadata=["distance"],
            include_vector=True,
        )
        recs = [_obj_to_rec(o) for o in (resp.objects or [])]
        if mode == "hybrid_rerank":
            recs = _rerank(vector, recs)
        return recs[:k]

    if mode == "production":
        recs = []
        if k_s:
            resp = coll.query.hybrid(
                query=query,
                vector=vector,
                alpha=config.HYBRID_ALPHA,
                limit=max(k_s * config.FETCH_MULTIPLIER, config.MIN_FETCH),
                filters=_category_filter("cpc-sections"),
                return_properties=PROPS,
                return_metadata=["distance"],
                include_vector=True,
            )
            hits = [_obj_to_rec(o) for o in (resp.objects or [])]
            recs += (_rerank(vector, hits) if rerank else hits)[:k_s]
        if k_c:
            resp = coll.query.hybrid(
                query=query,
                vector=vector,
                alpha=config.HYBRID_ALPHA,
                limit=max(k_c * config.FETCH_MULTIPLIER, config.MIN_FETCH),
                filters=_category_filter("case-laws"),
                return_properties=PROPS,
                return_metadata=["distance"],
                include_vector=True,
            )
            hits = [_obj_to_rec(o) for o in (resp.objects or [])]
            recs += (_rerank(vector, hits) if rerank else hits)[:k_c]
        return recs

    raise ValueError(f"unknown mode {mode!r}")


MODES = ["bm25", "dense", "hybrid", "hybrid_rerank", "production"]
