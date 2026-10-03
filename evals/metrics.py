"""Ranking metrics for binary relevance, with a page-level variant."""

from __future__ import annotations

import math


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(gold & set(ranked[:k])) / len(gold)


def precision_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    return len(gold & set(ranked[:k])) / k


def mrr(ranked: list[str], gold: set[str]) -> float:
    for i, rid in enumerate(ranked, 1):
        if rid in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    dcg = 0.0
    for i, rid in enumerate(ranked[:k], 1):
        if rid in gold:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def evaluate(ranked: list[str], gold: set[str]) -> dict:
    return {
        "recall@1": recall_at_k(ranked, gold, 1),
        "recall@5": recall_at_k(ranked, gold, 5),
        "recall@10": recall_at_k(ranked, gold, 10),
        "precision@5": precision_at_k(ranked, gold, 5),
        "mrr": mrr(ranked, gold),
        "ndcg@10": ndcg_at_k(ranked, gold, 10),
    }


def aggregate(rows: list[dict]) -> dict:
    """Mean of per-query metric dicts, preserving keys."""
    if not rows:
        return {}
    keys = rows[0].keys()
    n = len(rows)
    return {k: sum(r[k] for r in rows) / n for k in keys}


def page_keys(recs: list[dict]) -> list[str]:
    """Ordered, de-duplicated (source, page) keys from ranked chunk records."""
    return _keys(recs, lambda r: f"{r.get('source')}#{r.get('page')}")


def doc_keys(recs: list[dict]) -> list[str]:
    """Ordered, de-duplicated source keys from ranked chunk records."""
    return _keys(recs, lambda r: str(r.get("source")))


def _keys(recs, fn) -> list[str]:
    seen, out = set(), []
    for r in recs:
        key = fn(r)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out
