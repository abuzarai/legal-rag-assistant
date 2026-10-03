"""Assemble candidates into the committed eval set.

- drops unusable candidates and near-duplicate questions
- computes gold chunk sets by exact containment of the generator's verbatim
  support quote (so overlapping/adjacent chunks that carry the same passage are
  all marked relevant), plus the anchor chunk
- computes gold page keys for the secondary page-level metric
- writes evals/data/eval_set.jsonl and a review file for manual verification
"""

from __future__ import annotations

import json
import re
from collections import Counter

from evals import config


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a | b) else 0.0


def main() -> None:
    corpus = config.load_corpus()
    by_id = {c["chunk_id"]: c for c in corpus}
    candidates = [
        json.loads(l)
        for l in (config.DATA / "candidates.jsonl").read_text().splitlines()
        if l.strip()
    ]
    usable = [c for c in candidates if c.get("usable")]

    # --- de-duplicate questions (keep first) ---
    kept, dropped = [], []
    for c in usable:
        t = tokens(c["question"])
        if any(jaccard(t, tokens(k["question"])) >= 0.85 for k in kept):
            dropped.append({"anchor": c["anchor_chunk_id"], "q": c["question"]})
            continue
        kept.append(c)

    # --- build gold sets ---
    items = []
    for i, c in enumerate(kept, 1):
        quote = norm(c.get("support_quote", ""))
        gold_ids = {c["anchor_chunk_id"]}
        if len(quote) >= 60:
            for chunk in corpus:
                if quote in norm(chunk["content"]):
                    gold_ids.add(chunk["chunk_id"])
        gold = [by_id[g] for g in gold_ids if g in by_id]
        page_keys = sorted({f"{g['source']}#{g['page']}" for g in gold})
        q_tokens, quote_tokens = (
            tokens(c["question"]),
            tokens(c.get("support_quote", "")),
        )
        items.append(
            {
                "id": f"q{i:03d}",
                "question": c["question"],
                "question_type": c["question_type"],
                "reference_answer": c["reference_answer"],
                "citation": c["citation"],
                "anchor_chunk_id": c["anchor_chunk_id"],
                "anchor_source": c["anchor_source"],
                "anchor_page": c["anchor_page"],
                "anchor_category": c["anchor_category"],
                "gold_chunk_ids": sorted(gold_ids),
                "n_gold_chunks": len(gold_ids),
                "gold_page_keys": page_keys,
                "support_quote": c.get("support_quote", ""),
                "question_overlap_with_quote": round(
                    jaccard(q_tokens, quote_tokens), 3
                ),
                "provenance": f"llm-generated from anchor chunk; gold=anchor+quote-containment ({config.GEN_MODEL})",
                "manual_verdict": None,
            }
        )

    # Re-apply the manual review so re-running assembly is reproducible:
    # verdicts and corrected citations live in labels/manual_review.json.
    review_path = config.LABELS / "manual_review.json"
    if review_path.exists():
        review_items = json.loads(review_path.read_text()).get("items", {})
        for it in items:
            entry = review_items.get(it["id"])
            if not entry:
                continue
            it["manual_verdict"] = entry.get("verdict")
            if entry.get("corrected_citation"):
                it["citation"] = entry["corrected_citation"]

    out = config.EVAL_SET
    with out.open("w") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")

    # --- review file for the manual pass ---
    review = config.DATA / "review.jsonl"
    with review.open("w") as fh:
        for it in items:
            anchor = by_id[it["anchor_chunk_id"]]["content"]
            fh.write(
                json.dumps(
                    {
                        "id": it["id"],
                        "question": it["question"],
                        "reference_answer": it["reference_answer"],
                        "citation": it["citation"],
                        "anchor_source": it["anchor_source"],
                        "anchor_page": it["anchor_page"],
                        "anchor_excerpt": anchor[:700],
                    }
                )
                + "\n"
            )

    cats = Counter(it["anchor_category"] for it in items)
    multi = sum(1 for it in items if it["n_gold_chunks"] > 1)
    leaky = sum(1 for it in items if it["question_overlap_with_quote"] > 0.6)
    print(f"usable candidates: {len(usable)}")
    print(f"dropped as duplicate: {len(dropped)}")
    print(f"eval items: {len(items)}  by category: {dict(cats)}")
    print(f"items with >1 gold chunk: {multi}")
    print(f"items with question/support-quote token overlap >0.6: {leaky}")
    print(f"wrote {out}")
    print(f"wrote {review}")
    (config.RESULTS / "assemble_stats.json").write_text(
        json.dumps(
            {
                "usable_candidates": len(usable),
                "dropped_duplicates": dropped,
                "eval_items": len(items),
                "by_category": dict(cats),
                "multi_gold_items": multi,
                "leaky_items": leaky,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
