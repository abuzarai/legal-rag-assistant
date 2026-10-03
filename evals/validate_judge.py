"""Validate the LLM judge against a human-labeled sample.

python -m evals.validate_judge --dump 16      # write review file
python -m evals.validate_judge --score        # agreement + Cohen's kappa
"""

from __future__ import annotations

import argparse
import json
import random

from evals import config

HUMAN_LABELS = config.LABELS / "judge_human_labels.json"


def load(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa = sum(a) / n
    pb = sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=int, default=0)
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()

    config.ensure_import_path()
    from langchain_core.documents import Document

    from src.backend import rag

    judged = load(config.RESULTS / "judged.jsonl")
    gen = {
        f"{r['system']}:{r['id']}": r for r in load(config.RESULTS / "generators.jsonl")
    }

    if args.dump:
        rng = random.Random(5)
        by_sys: dict[str, list] = {}
        for r in judged:
            by_sys.setdefault(r["system"], []).append(r)
        sample = []
        per = max(1, args.dump // max(1, len(by_sys)))
        for rows in by_sys.values():
            rng.shuffle(rows)
            sample += rows[:per]
        out = config.LABELS / "judge_validation_review.jsonl"
        with out.open("w") as fh:
            for r in sample:
                g = gen.get(f"{r['system']}:{r['id']}", {})
                context = rag._join_context(
                    [
                        Document(
                            page_content=d["content"],
                            metadata={"source": d["source"], "page": d["page"]},
                        )
                        for d in g.get("retrieved", [])
                    ]
                )[:5000]
                fh.write(
                    json.dumps(
                        {
                            "key": r["key"],
                            "system": r["system"],
                            "question": g.get("question"),
                            "reference_answer": g.get("reference_answer"),
                            "context": context,
                            "answer": g.get("answer"),
                            "judge": r.get("verdict"),
                        }
                    )
                    + "\n"
                )
        print(f"wrote {out} ({len(sample)} items)")
        return

    if args.score:
        humans = json.loads(HUMAN_LABELS.read_text())
        rows = {r["key"]: r for r in judged}
        dims = {
            "grounded": ("faithfulness", "grounded"),
            "citation_accurate": ("citation_accuracy", "accurate"),
            "hallucination": ("hallucination", "present"),
        }
        report = {}
        for name, (section, field) in dims.items():
            h, j = [], []
            for key, hlabel in humans.items():
                r = rows.get(key)
                if not r or not r.get("verdict"):
                    continue
                jlabel = bool(r["verdict"][section][field])
                h.append(bool(hlabel[name]))
                j.append(jlabel)
            if h:
                agree = sum(x == y for x, y in zip(h, j)) / len(h)
                report[name] = {"n": len(h), "agreement": agree, "kappa": kappa(h, j)}
        # correctness collapsed to correct vs not
        h, j = [], []
        for key, hlabel in humans.items():
            r = rows.get(key)
            if not r or not r.get("verdict"):
                continue
            h.append(bool(hlabel.get("correct")))
            j.append(r["verdict"]["answer_correctness"]["verdict"] == "correct")
        if h:
            report["correct"] = {
                "n": len(h),
                "agreement": sum(x == y for x, y in zip(h, j)) / len(h),
                "kappa": kappa(h, j),
            }
        (config.RESULTS / "judge_validation.json").write_text(
            json.dumps(report, indent=2)
        )
        print(json.dumps(report, indent=2))
        return

    print("nothing to do; pass --dump N or --score")


if __name__ == "__main__":
    main()
