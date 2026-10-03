"""Run the retrieval ablation and write per-query + aggregate metrics.

python -m evals.run_retrieval
"""

from __future__ import annotations

import json
import time
from collections import defaultdict

from evals import config, metrics, retrieval

CONFIGS = {
    "bm25": {"mode": "bm25", "k": 10},
    "dense": {"mode": "dense", "k": 10},
    "hybrid": {"mode": "hybrid", "k": 10},
    "hybrid_rerank": {"mode": "hybrid_rerank", "k": 10},
    "production@10": {"mode": "production", "k_s": 6, "k_c": 4},
    "production@5": {"mode": "production", "k_s": 3, "k_c": 2},
    "production@5 +rerank": {
        "mode": "production",
        "k_s": 3,
        "k_c": 2,
        "rerank": True,
    },
}


def main() -> None:
    src = config.init_credentials("embed")
    config.ensure_import_path()
    print(f"embed key: {src}, model={config.EMBEDDING_MODEL}")

    items = config.load_eval_set()
    print(f"eval items: {len(items)}")

    t0 = time.time()
    questions = [it["question"] for it in items]
    vectors = retrieval.embed_queries(questions)
    print(f"embedded {len(vectors)} queries in {time.time() - t0:.1f}s")

    rankings_path = config.RESULTS / "retrieval_rankings.jsonl"
    per_query: dict[str, list[dict]] = defaultdict(list)
    t1 = time.time()

    with rankings_path.open("w") as fh:
        for ci, (name, cfg) in enumerate(CONFIGS.items(), 1):
            for n, it in enumerate(items, 1):
                recs = retrieval.run_config(
                    it["question"], vectors[it["question"]], **cfg
                )
                ranked_ids = [r["chunk_id"] for r in recs]
                ranked_pages = metrics.page_keys(recs)
                ranked_docs = metrics.doc_keys(recs)
                gold_ids = set(it["gold_chunk_ids"])
                gold_pages = set(it["gold_page_keys"])
                gold_docs = {k.split("#")[0] for k in it["gold_page_keys"]}
                row = {
                    "config": name,
                    "id": it["id"],
                    "category": it["anchor_category"],
                    "question_type": it["question_type"],
                    "ranked_ids": ranked_ids,
                    "ranked_pages": ranked_pages,
                    "chunk": metrics.evaluate(ranked_ids, gold_ids),
                    "page": metrics.evaluate(ranked_pages, gold_pages),
                    "doc": metrics.evaluate(ranked_docs, gold_docs),
                }
                per_query[name].append(row)
                fh.write(
                    json.dumps(
                        {
                            "config": name,
                            "id": it["id"],
                            "question": it["question"],
                            "ranked_ids": ranked_ids,
                            "gold_ids": sorted(gold_ids),
                        }
                    )
                    + "\n"
                )
            print(f"[{ci}/{len(CONFIGS)}] {name} done ({time.time() - t1:.1f}s)")

    results = {}
    for name in CONFIGS:
        rows = per_query[name]
        entry = {
            "chunk_level": metrics.aggregate([r["chunk"] for r in rows]),
            "page_level": metrics.aggregate([r["page"] for r in rows]),
            "doc_level": metrics.aggregate([r["doc"] for r in rows]),
            "n": len(rows),
        }
        for cat in ("cpc-sections", "case-laws"):
            sub = [r for r in rows if r["category"] == cat]
            if sub:
                entry[f"chunk_level::{cat}"] = metrics.aggregate(
                    [r["chunk"] for r in sub]
                )
                entry[f"page_level::{cat}"] = metrics.aggregate(
                    [r["page"] for r in sub]
                )
                entry[f"n::{cat}"] = len(sub)
        results[name] = entry

    out = config.RESULTS / "retrieval.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}  ({time.time() - t1:.1f}s)")

    hdr = f"{'config':16s} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'P@5':>6} {'MRR':>6} {'NDCG@10':>8}"
    for level in ("chunk_level", "page_level", "doc_level"):
        print(f"\n=== {level} ===")
        print(hdr)
        for name, e in results.items():
            m = e[level]
            print(
                f"{name:16s} {m['recall@1']:6.3f} {m['recall@5']:6.3f} {m['recall@10']:6.3f} "
                f"{m['precision@5']:6.3f} {m['mrr']:6.3f} {m['ndcg@10']:8.3f}"
            )
    print("\n=== chunk_level by category ===")
    for cat in ("cpc-sections", "case-laws"):
        print(f"-- {cat} --")
        print(hdr)
        for name, e in results.items():
            m = e.get(f"chunk_level::{cat}")
            if not m:
                continue
            print(
                f"{name:16s} {m['recall@1']:6.3f} {m['recall@5']:6.3f} {m['recall@10']:6.3f} "
                f"{m['precision@5']:6.3f} {m['mrr']:6.3f} {m['ndcg@10']:8.3f}"
            )


if __name__ == "__main__":
    main()
