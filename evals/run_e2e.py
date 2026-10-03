"""End-to-end latency: production retrieval + a runnable generator, N runs.

Uses gemini-3.5-flash-lite because the production gemini-2.5-flash quota was
constrained (see REPORT.md). The retrieval half is identical to production.

    python -m evals.run_e2e --runs 30
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from evals import config, llm, retrieval
from evals.run_generation import PRODUCTION_SYSTEM_PROMPT
from evals.run_generators import HUMAN_TEMPLATE


def pct(xs, p):
    return float(np.percentile(xs, p)) if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    args = ap.parse_args()

    config.init_credentials("embed")
    config.ensure_import_path()
    from langchain_core.documents import Document

    from src.backend import rag

    items = config.load_eval_set()[: args.runs]
    rows = []
    out = config.RESULTS / "e2e_latency.jsonl"
    with out.open("w") as fh:
        for i, it in enumerate(items, 1):
            t0 = time.perf_counter()
            vec = retrieval.embed_queries([it["question"]])[it["question"]]
            t_embed = time.perf_counter() - t0
            t1 = time.perf_counter()
            recs = retrieval.run_config(it["question"], vec, mode="production")
            t_search = time.perf_counter() - t1
            docs = [
                Document(
                    page_content=r["content"],
                    metadata={"source": r["source"], "page": r["page"]},
                )
                for r in recs
            ]
            prompt = HUMAN_TEMPLATE.format(
                context=rag._join_context(docs), question=it["question"]
            )
            t2 = time.perf_counter()
            try:
                _, in_tok, out_tok = llm.generate_text(
                    "gen", args.model, prompt, system=PRODUCTION_SYSTEM_PROMPT
                )
                err = None
            except Exception as exc:  # noqa: BLE001
                in_tok = out_tok = 0
                err = str(exc)[:200]
            t_gen = time.perf_counter() - t2
            row = {
                "id": it["id"],
                "model": args.model,
                "embed_s": t_embed,
                "search_s": t_search,
                "generation_s": t_gen,
                "e2e_s": t_embed + t_search + t_gen,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "error": err,
            }
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            print(f"[{i}/{len(items)}] e2e={row['e2e_s']:.2f}s err={err}")

    ok = [r for r in rows if not r["error"]]
    summary = {
        "model": args.model,
        "n": len(ok),
        "embed_ms": {
            f"p{p}": pct([r["embed_s"] for r in ok], p) * 1000 for p in (50, 95, 99)
        },
        "search_ms": {
            f"p{p}": pct([r["search_s"] for r in ok], p) * 1000 for p in (50, 95, 99)
        },
        "generation_ms": {
            f"p{p}": pct([r["generation_s"] for r in ok], p) * 1000
            for p in (50, 95, 99)
        },
        "e2e_ms": {
            f"p{p}": pct([r["e2e_s"] for r in ok], p) * 1000 for p in (50, 95, 99)
        },
    }
    (config.RESULTS / "e2e_latency_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
