"""Generate candidate eval questions grounded in real corpus passages.

Each candidate is anchored to one chunk from the live index. The anchor chunk
is the primary ground-truth positive; the generator must also return a verbatim
support quote so we can find any other chunk that contains the same passage.

Resumable: re-running skips anchors already present in candidates.jsonl.

    python -m evals.gen_questions --sample 6
    python -m evals.gen_questions --total 140
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from evals import config, llm


def allocate(by_doc: dict[str, list[dict]], total: int) -> dict[str, int]:
    """Allocate the budget: equal split across categories, then sqrt(chunk
    count) within a category.

    The production index is 93% CPC text / 7% case-law by chunk count, but the
    system always pools from both categories (3 CPC + 2 case-law chunks). A
    proportional sample would leave the case-law pool barely measured, so the
    budget is split evenly across the two categories and metrics are also
    reported per category.
    """
    from collections import defaultdict

    cat_docs: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    for doc, chunks in by_doc.items():
        cat_docs[chunks[0]["category"]][doc] = chunks

    alloc: dict[str, int] = {}
    cats = sorted(cat_docs)
    per = total // len(cats)
    for i, cat in enumerate(cats):
        n = per + (total - per * len(cats) if i == 0 else 0)
        weights = {d: len(c) ** 0.5 for d, c in cat_docs[cat].items()}
        scale = n / sum(weights.values())
        for doc, w in weights.items():
            alloc[doc] = max(2, round(w * scale))
    return alloc


def sample_anchors(by_doc, alloc, rng) -> list[dict]:
    anchors = []
    for doc, n in alloc.items():
        chunks = by_doc[doc]
        usable = [c for c in chunks if len(c["content"].strip()) >= 350] or chunks
        rng.shuffle(usable)
        multi_page = len({c["page"] for c in usable}) > 1
        picked, seen_pages = [], defaultdict(int)
        for c in usable:
            if len(picked) >= n:
                break
            if multi_page and seen_pages[c["page"]] >= 1 and len(usable) > n * 2:
                continue
            seen_pages[c["page"]] += 1
            picked.append(c)
        for c in usable:
            if len(picked) >= n:
                break
            if c not in picked:
                picked.append(c)
        anchors.extend(picked)
    return anchors


SYSTEM = """You build evaluation sets for a Pakistani legal RAG system.
You are given ONE verbatim passage from the system's corpus. Produce a single
question that the passage answers, plus a reference answer that a careful
lawyer would accept.

Rules:
- The question must be answerable ENTIRELY from the passage.
- Paraphrase: do not copy long phrases from the passage into the question.
- The question must read like something a Pakistani litigant or junior lawyer
  would actually ask (never mention "the passage" or "the text above").
- reference_answer: 1-3 sentences, using only facts stated in the passage.
- citation: the legal provision or case. For a statute passage use the CPC
  provision (e.g. "Order VII, Rule 11, CPC"). For a judgment use the case
  citation that appears in the passage (e.g. "PLD 2023 Supreme Court 174")
  plus the provision discussed if one is named.
- support_quote: an EXACT substring of the passage (at least 80 characters,
  copied character-for-character) that by itself supports the answer.
- If the passage is OCR noise, a bare citation list, or otherwise cannot
  support a real question, set "usable": false and leave other fields empty.
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "usable": {"type": "boolean"},
        "question": {"type": "string"},
        "question_type": {
            "type": "string",
            "enum": [
                "definitional",
                "procedural",
                "consequence",
                "comparative",
                "scenario",
                "holding",
            ],
        },
        "reference_answer": {"type": "string"},
        "citation": {"type": "string"},
        "support_quote": {"type": "string"},
    },
    "required": ["usable", "question", "question_type", "reference_answer", "citation"],
}


def build_prompt(chunk: dict) -> str:
    return (
        f"Source document: {chunk['source']}\n"
        f"Page: {chunk['page']}\n"
        f"Category: {chunk['category']}\n\n"
        "<passage>\n" + chunk["content"] + "\n</passage>"
    )


def generate_one(chunk: dict) -> dict:
    data, in_tok, out_tok = llm.generate_json(
        "gen", config.GEN_MODEL, build_prompt(chunk), RESPONSE_SCHEMA, system=SYSTEM
    )
    return {
        "anchor_chunk_id": chunk["chunk_id"],
        "anchor_source": chunk["source"],
        "anchor_page": chunk["page"],
        "anchor_category": chunk["category"],
        "anchor_content": chunk["content"],
        "usable": data.get("usable", False),
        "question": (data.get("question") or "").strip(),
        "question_type": data.get("question_type"),
        "reference_answer": (data.get("reference_answer") or "").strip(),
        "citation": (data.get("citation") or "").strip(),
        "support_quote": (data.get("support_quote") or "").strip(),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
    }


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def load_existing(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--total", type=int, default=140)
    ap.add_argument("--seed", type=int, default=1908)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    config.ensure_import_path()
    print(f"gen model={config.GEN_MODEL} key_pool=gen")

    corpus = config.load_corpus()
    by_doc = defaultdict(list)
    for c in corpus:
        by_doc[c["source"]].append(c)

    rng = random.Random(args.seed)
    alloc = allocate(by_doc, args.total)
    anchors = sample_anchors(by_doc, alloc, rng)
    if args.sample:
        anchors = anchors[: args.sample]

    out = config.DATA / "candidates.jsonl"
    existing = load_existing(out)
    done = {r["anchor_chunk_id"] for r in existing}
    todo = [a for a in anchors if a["chunk_id"] not in done]
    print(f"anchors={len(anchors)} already_done={len(done)} todo={len(todo)}")

    results = list(existing)
    failures = []
    t0 = time.time()

    def work(a):
        rec = generate_one(a)
        if (
            rec["usable"]
            and rec["support_quote"]
            and norm(rec["support_quote"]) not in norm(rec["anchor_content"])
        ):
            rec["usable"] = False
            rec["reject_reason"] = "support_quote not verbatim in anchor"
        return rec

    if todo:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = {pool.submit(work, a): a for a in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                a = futs[fut]
                try:
                    rec = fut.result()
                except Exception as exc:  # noqa: BLE001
                    failures.append(
                        {"anchor_chunk_id": a["chunk_id"], "error": str(exc)[:300]}
                    )
                    print(
                        f"[{i}/{len(todo)}] FAIL {a['chunk_id'][:8]}: {str(exc)[:110]}"
                    )
                    continue
                results.append(rec)
                status = "ok" if rec["usable"] else "unusable"
                print(
                    f"[{i}/{len(todo)}] {status} {a['chunk_id'][:8]} {rec['question'][:70]!r}"
                )
        # persist incrementally after the pool drains
        with out.open("w") as fh:
            for rec in results:
                fh.write(json.dumps(rec) + "\n")

    usable = [r for r in results if r["usable"]]
    by_cat = defaultdict(int)
    for r in usable:
        by_cat[r["anchor_category"]] += 1
    failures_path = config.RESULTS / "gen_failures.json"
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path.write_text(json.dumps(failures, indent=2))
    in_tok = sum(r.get("input_tokens", 0) for r in results)
    out_tok = sum(r.get("output_tokens", 0) for r in results)
    print(
        f"\nusable {len(usable)}/{len(results)} failures={len(failures)} "
        f"by_category={dict(by_cat)} elapsed={time.time() - t0:.1f}s "
        f"tokens_in={in_tok} tokens_out={out_tok}"
    )
    (config.RESULTS / "gen_usage.json").write_text(
        json.dumps(
            {
                "model": config.GEN_MODEL,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
            },
            indent=2,
        )
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
