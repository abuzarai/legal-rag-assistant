"""Retrieval-only latency, measured the way production does it.

Each run: one query embedding (network) + production statute-first retrieval
(two hybrid searches with local rerank). No LLM. N is stated in the output.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from evals import config, retrieval


def pct(xs, p):
    return float(np.percentile(xs, p)) if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--questions", type=int, default=40)
    ap.add_argument("--pace", type=float, default=0.6)
    args = ap.parse_args()

    src = config.init_credentials("embed")
    config.ensure_import_path()
    print(f"embed key: {src}")

    items = config.load_eval_set()[: args.questions]
    out = config.RESULTS / "latency.jsonl"
    rows = []
    with out.open("w") as fh:
        for rep in range(args.repeats):
            for it in items:
                q = it["question"]
                t0 = time.perf_counter()
                vec = retrieval.embed_queries([q])[q]
                t_embed = time.perf_counter() - t0
                t1 = time.perf_counter()
                recs = retrieval.run_config(q, vec, mode="production")
                t_search = time.perf_counter() - t1
                row = {
                    "id": it["id"],
                    "repeat": rep,
                    "embed_s": t_embed,
                    "search_s": t_search,
                    "retrieval_s": t_embed + t_search,
                    "n_returned": len(recs),
                }
                rows.append(row)
                fh.write(json.dumps(row) + "\n")
                time.sleep(args.pace)
            print(f"repeat {rep + 1}/{args.repeats} done")

    for k in ("embed_s", "search_s", "retrieval_s"):
        xs = [r[k] for r in rows]
        print(
            f"{k:12s} n={len(xs)} p50={pct(xs, 50) * 1000:.1f}ms "
            f"p95={pct(xs, 95) * 1000:.1f}ms p99={pct(xs, 99) * 1000:.1f}ms"
        )
    (config.RESULTS / "latency_summary.json").write_text(
        json.dumps(
            {
                "n": len(rows),
                "repeats": args.repeats,
                "questions": args.questions,
                "embed_ms": {
                    f"p{p}": pct([r["embed_s"] for r in rows], p) * 1000
                    for p in (50, 95, 99)
                },
                "search_ms": {
                    f"p{p}": pct([r["search_s"] for r in rows], p) * 1000
                    for p in (50, 95, 99)
                },
                "retrieval_ms": {
                    f"p{p}": pct([r["retrieval_s"] for r in rows], p) * 1000
                    for p in (50, 95, 99)
                },
            },
            indent=2,
        )
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
