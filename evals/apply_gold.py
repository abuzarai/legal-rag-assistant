"""Merge pooled gold labels into eval_set.jsonl once expand_gold has run.

python -m evals.apply_gold
python -m evals.apply_gold --spotcheck 15
"""

from __future__ import annotations

import argparse
import json

from evals import config


def load(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spotcheck", type=int, default=0)
    args = ap.parse_args()

    items = load(config.EVAL_SET)
    labels = {r["id"]: r for r in load(config.DATA / "gold_labels.jsonl")}
    corpus = {c["chunk_id"]: c for c in config.load_corpus()}

    n_expanded = 0
    for it in items:
        lab = labels.get(it["id"])
        if not lab:
            continue
        gold = set(lab["gold_chunk_ids"])
        gold.add(it["anchor_chunk_id"])
        it["gold_chunk_ids"] = sorted(gold)
        it["n_gold_chunks"] = len(gold)
        it["gold_page_keys"] = sorted(
            {f"{corpus[c]['source']}#{corpus[c]['page']}" for c in gold if c in corpus}
        )
        it["pool_size"] = lab["pool_size"]
        it["llm_relevant_added"] = sorted(
            set(lab["llm_relevant"]) - {it["anchor_chunk_id"]}
        )
        it["gold_source"] = (
            "anchor + verbatim-quote containment + pooled LLM relevance (manually spot-checked)"
        )
        if len(gold) > 1:
            n_expanded += 1

    with config.EVAL_SET.open("w") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")

    sizes = [it["n_gold_chunks"] for it in items]
    print(
        f"items={len(items)} items_with_multiple_gold={n_expanded} "
        f"mean_gold={sum(sizes) / len(sizes):.2f} max_gold={max(sizes)}"
    )

    if args.spotcheck:
        import random

        rng = random.Random(11)
        sample = [it for it in items if it.get("llm_relevant_added")]
        rng.shuffle(sample)
        for it in sample[: args.spotcheck]:
            print("=" * 90)
            print(it["id"], "|", it["question"])
            print("reference:", it["reference_answer"])
            for cid in it["llm_relevant_added"][:2]:
                c = corpus[cid]
                print(
                    f"  +ADDED ({c['source'].split('/')[-1]}): {c['content'][:380]!r}"
                )


if __name__ == "__main__":
    main()
