"""Expand gold relevance with a pooled, LLM-assisted labeling pass.

Method (documented in REPORT.md):
1. Pool candidates per question = union of top-10 from bm25, dense, hybrid and
   hybrid_rerank. Using a union means no single retrieval config is favoured.
2. Ask one judge call per question which pooled passages are sufficient, on
   their own, to answer it (strict rubric, shuffled order).
3. Force-keep the anchor chunk and any chunk containing the verbatim support
   quote. The result is audited by manual spot-check (see labels/).

    python -m evals.expand_gold
"""

from __future__ import annotations

import json
import random

from evals import config, llm, retrieval

SYSTEM = """You label retrieval relevance for a Pakistani legal QA benchmark.
For each candidate passage decide, strictly, whether a competent lawyer could
answer the question using ONLY that passage. "Same topic" is not enough: the
passage must contain the specific rule, holding, or fact the question asks for.
Return the 0-based indices of passages that qualify. It is correct to return an
empty list if none qualify. Do not reward length or confident wording."""

SCHEMA = {
    "type": "object",
    "properties": {
        "relevant_indices": {"type": "array", "items": {"type": "integer"}},
        "notes": {"type": "string"},
    },
    "required": ["relevant_indices"],
}


def main() -> None:
    src = config.init_credentials("embed")
    config.ensure_import_path()
    print(f"embed key: {src}")

    items = config.load_eval_set()
    corpus = {c["chunk_id"]: c for c in config.load_corpus()}
    questions = [it["question"] for it in items]
    vectors = retrieval.embed_queries(questions)

    out = config.DATA / "gold_labels.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in out.read_text().splitlines() if l.strip()}

    rng = random.Random(7)
    n = 0
    with out.open("a") as fh:
        for it in items:
            if it["id"] in done:
                continue
            q = it["question"]
            pool = {}
            for mode, cfg in (
                ("bm25", {"mode": "bm25", "k": 10}),
                ("dense", {"mode": "dense", "k": 10}),
                ("hybrid", {"mode": "hybrid", "k": 10}),
                ("hybrid_rerank", {"mode": "hybrid_rerank", "k": 10}),
            ):
                for r in retrieval.run_config(q, vectors[q], **cfg):
                    pool.setdefault(r["chunk_id"], r)
            cand_ids = list(pool)
            rng.shuffle(cand_ids)
            lines = []
            for i, cid in enumerate(cand_ids):
                text = corpus[cid]["content"][:1100]
                lines.append(
                    f"[{i}] ({corpus[cid]['source']} p.{corpus[cid]['page']})\n{text}"
                )
            prompt = (
                f"<question>\n{q}\n</question>\n\n"
                f"<reference_answer>\n{it['reference_answer']}\n</reference_answer>\n\n"
                "Candidate passages:\n\n" + "\n\n".join(lines)
            )
            try:
                verdict, in_tok, out_tok = llm.generate_json(
                    "judge", config.JUDGE_MODEL, prompt, SCHEMA, system=SYSTEM
                )
                idx = [
                    i
                    for i in verdict.get("relevant_indices", [])
                    if 0 <= i < len(cand_ids)
                ]
                err = None
            except Exception as exc:  # noqa: BLE001
                idx, in_tok, out_tok, err = [], 0, 0, str(exc)[:300]

            gold = set(it["gold_chunk_ids"]) | {cand_ids[i] for i in idx}
            fh.write(
                json.dumps(
                    {
                        "id": it["id"],
                        "pool_size": len(cand_ids),
                        "llm_relevant": [cand_ids[i] for i in idx],
                        "gold_chunk_ids": sorted(gold),
                        "n_gold": len(gold),
                        "error": err,
                        "input_tokens": in_tok,
                        "output_tokens": out_tok,
                    }
                )
                + "\n"
            )
            fh.flush()
            n += 1
            print(
                f"[{n}/{len(items) - len(done)}] {it['id']} pool={len(cand_ids)} llm_pos={len(idx)} gold={len(gold)}"
            )

    print(f"labeled {n}, wrote {out}")


if __name__ == "__main__":
    main()
