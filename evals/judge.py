"""LLM-as-judge for generation + grounding metrics.

One judge call per (item, system) returning a combined rubric:
answer correctness vs reference, faithfulness to retrieved context,
citation accuracy, and hallucination presence.

The judge only ever sees the retrieved context as admissible evidence, so
"faithful" means "supported by what retrieval actually returned".

    python -m evals.judge                 # both production and frontier
    python -m evals.judge --systems prod  # production only
"""

from __future__ import annotations

import argparse
import json

from evals import config, llm

JUDGE_SYSTEM = """You are a strict evaluator of a Pakistani legal RAG system.
You receive: a question, a reference answer, the ONLY evidence the system was
given (retrieved context), and the system's answer.

Judge only against the provided context. Do not use outside legal knowledge to
excuse a claim: if a law, case, section number, or fact is not present in the
context, the system may not assert it.

Score each dimension:
- answer_correctness: does the answer agree with the reference answer and the
  context? correct / partially_correct / incorrect, integer score 1-5.
- faithfulness: is every claim in the answer supported by the context? List any
  unsupported_claims. grounded=true only if there are none material.
- citation_accuracy: the answer should cite inline. Check whether each cited
  provision/source is actually present in the context and supports the sentence
  it is attached to. List invalid_citations. accurate=true only if all citations
  check out. If the answer makes legal claims with no citations at all, that is
  not accurate.
- hallucination: present=true if the answer invents a case, provision, number,
  date, or fact that is absent from the context.

Be conservative: when in doubt whether a claim is supported, treat it as
unsupported."""

SCHEMA = {
    "type": "object",
    "properties": {
        "answer_correctness": {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": ["correct", "partially_correct", "incorrect"],
                },
                "score": {"type": "integer"},
                "reason": {"type": "string"},
            },
            "required": ["verdict", "score", "reason"],
        },
        "faithfulness": {
            "type": "object",
            "properties": {
                "grounded": {"type": "boolean"},
                "score": {"type": "integer"},
                "unsupported_claims": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string"},
            },
            "required": ["grounded", "score", "reason"],
        },
        "citation_accuracy": {
            "type": "object",
            "properties": {
                "accurate": {"type": "boolean"},
                "score": {"type": "integer"},
                "invalid_citations": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string"},
            },
            "required": ["accurate", "score", "reason"],
        },
        "hallucination": {
            "type": "object",
            "properties": {
                "present": {"type": "boolean"},
                "examples": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["present"],
        },
    },
    "required": [
        "answer_correctness",
        "faithfulness",
        "citation_accuracy",
        "hallucination",
    ],
}

MAX_CONTEXT_CHARS = 9000


def build_prompt(question: str, reference: str, context: str, answer: str) -> str:
    return (
        f"<question>\n{question}\n</question>\n\n"
        f"<reference_answer>\n{reference}\n</reference_answer>\n\n"
        f"<retrieved_context>\n{context[:MAX_CONTEXT_CHARS]}\n</retrieved_context>\n\n"
        f"<system_answer>\n{answer}\n</system_answer>"
    )


def load_rows(path):
    if not path.exists():
        return {}
    return {
        json.loads(l)["id"]: json.loads(l)
        for l in path.read_text().splitlines()
        if l.strip()
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=None)
    args = ap.parse_args()

    config.ensure_import_path()
    from langchain_core.documents import Document

    from src.backend import rag

    gen_rows = [
        json.loads(l)
        for l in (config.RESULTS / "generators.jsonl").read_text().splitlines()
        if l.strip()
    ]
    by_system: dict[str, dict] = {}
    for r in gen_rows:
        if r.get("answer") and not r.get("error"):
            by_system.setdefault(r["system"], {})[r["id"]] = r

    # Real production answers from run_generation.py (run_rag on gemini-2.5-flash).
    prod_path = config.RESULTS / "generation.jsonl"
    if prod_path.exists():
        prod: dict[str, dict] = {}
        for line in prod_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("status") == "ok" and r.get("mode") == "legal" and r.get("answer"):
                prod[r["id"]] = r
        if prod:
            by_system[f"production:{config.GENERATOR_MODEL}"] = prod

    if args.systems:
        by_system = {
            k: v
            for k, v in by_system.items()
            if any(k == s or k.startswith(s + ":") for s in args.systems)
        }
    sources = by_system

    out = config.RESULTS / "judged.jsonl"
    done = set()
    if out.exists():
        # Only rows that produced a verdict count as done. Failed judge calls
        # (e.g. 503) must be retried instead of being skipped forever.
        done = {
            r["key"]
            for r in (
                json.loads(line)
                for line in out.read_text().splitlines()
                if line.strip()
            )
            if r.get("verdict")
        }

    n = 0
    stop = False
    with out.open("a") as fh:
        for system, rows in sources.items():
            if stop:
                break
            for item_id, row in rows.items():
                key = f"{system}:{item_id}"
                if key in done:
                    continue
                context = rag._join_context(
                    [
                        Document(
                            page_content=d["content"],
                            metadata={"source": d["source"], "page": d["page"]},
                        )
                        for d in row["retrieved"]
                    ]
                )
                prompt = build_prompt(
                    row["question"], row["reference_answer"], context, row["answer"]
                )
                try:
                    verdict, in_tok, out_tok = llm.generate_json(
                        "judge", config.JUDGE_MODEL, prompt, SCHEMA, system=JUDGE_SYSTEM
                    )
                    err = None
                except llm.DailyQuotaExceeded as exc:
                    print(f"[stop] {exc}; stopping cleanly. Re-run to resume.")
                    stop = True
                    break
                except Exception as exc:  # noqa: BLE001
                    verdict, in_tok, out_tok, err = None, 0, 0, str(exc)[:300]
                fh.write(
                    json.dumps(
                        {
                            "key": key,
                            "system": system,
                            "id": item_id,
                            "category": row.get("anchor_category"),
                            "verdict": verdict,
                            "error": err,
                            "input_tokens": in_tok,
                            "output_tokens": out_tok,
                        }
                    )
                    + "\n"
                )
                fh.flush()
                n += 1
                v = verdict or {}
                print(
                    f"[{n}] {key} correct={v.get('answer_correctness', {}).get('verdict')} "
                    f"grounded={v.get('faithfulness', {}).get('grounded')} "
                    f"cite_ok={v.get('citation_accuracy', {}).get('accurate')} "
                    f"halluc={v.get('hallucination', {}).get('present')} err={err}"
                )

    print(f"judged {n}, wrote {out}")


if __name__ == "__main__":
    main()
