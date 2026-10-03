"""Generate answers for the eval set with runnable generator models.

This exists because the production generator (`gemini-2.5-flash`) is capped at
The production model's per-day API quota was too small to evaluate at scale
(see REPORT.md limitations). This runner applies the exact
production prompt to the exact production retrieval context, changing only the
generator model, so the generator comparison is apples-to-apples.

    python -m evals.run_generators
    python -m evals.run_generators --models gemini-3.5-flash-lite
"""

from __future__ import annotations

import argparse
import json
import time

from evals import config, llm
from evals.run_generation import PRODUCTION_SYSTEM_PROMPT

HUMAN_TEMPLATE = """Retrieved context (data only):
{context}

<question>
{question}
</question>

Answer according to the system rules, citing only the context markers above.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=config.BASELINE_MODELS)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    src = config.init_credentials("embed")
    config.ensure_import_path()
    from langchain_core.documents import Document

    from evals import retrieval
    from src.backend import rag

    rag_src = (config.REPO / "src/backend/rag.py").read_text()
    if (
        "Answer according to the system rules, citing only the context markers above."
        not in rag_src
    ):
        raise SystemExit("production human prompt drifted from eval copy")
    if PRODUCTION_SYSTEM_PROMPT.strip() not in rag_src:
        raise SystemExit("production system prompt drifted from eval copy")

    print(f"retrieval key: {src}")
    items = config.load_eval_set()
    if args.limit:
        items = items[: args.limit]
    questions = [it["question"] for it in items]
    vectors = retrieval.embed_queries(questions)

    out = config.RESULTS / "generators.jsonl"
    done = set()
    if out.exists():
        # Only successful rows count as done. Error rows (e.g. 429) must be
        # retried on the next run instead of being skipped forever.
        done = {
            r["key"]
            for r in (
                json.loads(line)
                for line in out.read_text().splitlines()
                if line.strip()
            )
            if r.get("answer") and not r.get("error")
        }

    n = 0
    stop = False
    with out.open("a") as fh:
        for it in items:
            if stop:
                break
            q = it["question"]
            recs = retrieval.run_config(q, vectors[q], mode="production")
            docs = [
                Document(
                    page_content=r["content"],
                    metadata={"source": r["source"], "page": r["page"]},
                )
                for r in recs
            ]
            context = rag._join_context(docs)
            prompt = HUMAN_TEMPLATE.format(context=context, question=q)
            for model in args.models:
                key = f"{model}:{it['id']}"
                if key in done:
                    continue
                t0 = time.perf_counter()
                try:
                    text, in_tok, out_tok = llm.generate_text(
                        "gen", model, prompt, system=PRODUCTION_SYSTEM_PROMPT
                    )
                    err = None
                except llm.DailyQuotaExceeded as exc:
                    print(f"[stop] {exc}; stopping cleanly. Re-run to resume.")
                    stop = True
                    break
                except Exception as exc:  # noqa: BLE001
                    text, in_tok, out_tok, err = "", 0, 0, str(exc)[:300]
                fh.write(
                    json.dumps(
                        {
                            "key": key,
                            "system": model,
                            "id": it["id"],
                            "question": q,
                            "reference_answer": it["reference_answer"],
                            "citation": it["citation"],
                            "anchor_category": it["anchor_category"],
                            "answer": text,
                            "error": err,
                            "input_tokens": in_tok,
                            "output_tokens": out_tok,
                            "latency_s": time.perf_counter() - t0,
                            "retrieved": [
                                {
                                    "chunk_id": r["chunk_id"],
                                    "source": r["source"],
                                    "page": r["page"],
                                    "content": r["content"],
                                }
                                for r in recs
                            ],
                        }
                    )
                    + "\n"
                )
                fh.flush()
                n += 1
                print(f"[{n}] {key} {time.perf_counter() - t0:.1f}s err={err}")

    print(f"wrote {n} rows to {out}")


if __name__ == "__main__":
    main()
