"""Aggregate judged results, latency, and list-price cost into summary.json."""

from __future__ import annotations

import json
from collections import Counter
from datetime import date

import numpy as np

from evals import config


def pct(xs, p):
    return float(np.percentile(xs, p)) if xs else float("nan")


def load(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def judge_metrics(rows):
    rows = [r for r in rows if r.get("verdict")]
    if not rows:
        return {"n": 0}
    v = [r["verdict"] for r in rows]
    ac = Counter(x["answer_correctness"]["verdict"] for x in v)
    return {
        "n": len(rows),
        "answer_correct_rate": ac.get("correct", 0) / len(v),
        "answer_partial_rate": ac.get("partially_correct", 0) / len(v),
        "answer_incorrect_rate": ac.get("incorrect", 0) / len(v),
        "answer_correctness_mean": float(
            np.mean([x["answer_correctness"]["score"] for x in v])
        ),
        "faithfulness_grounded_rate": float(
            np.mean([x["faithfulness"]["grounded"] for x in v])
        ),
        "faithfulness_mean": float(np.mean([x["faithfulness"]["score"] for x in v])),
        "citation_accuracy_rate": float(
            np.mean([x["citation_accuracy"]["accurate"] for x in v])
        ),
        "citation_accuracy_mean": float(
            np.mean([x["citation_accuracy"]["score"] for x in v])
        ),
        "hallucination_rate": float(
            np.mean([x["hallucination"]["present"] for x in v])
        ),
    }


def est_tokens(text):
    return max(1, round(len(text or "") / 4))


def model_cost(rows, model):
    """Cost/latency for substitute-generator rows (generators.jsonl)."""
    pr = config.PRICING.get(model, {"input": 0.0, "output": 0.0})
    in_tok = sum(r["input_tokens"] for r in rows)
    out_tok = sum(r["output_tokens"] for r in rows)
    usd = in_tok / 1e6 * pr["input"] + out_tok / 1e6 * pr["output"]
    lat = [r["latency_s"] for r in rows]
    return {
        "n": len(rows),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "usd_total": usd,
        "usd_per_query": usd / len(rows) if rows else float("nan"),
        "generation_ms": {f"p{p}": pct(lat, p) * 1000 for p in (50, 95, 99)},
    }


def production_cost(rows):
    """Cost/latency for real production rows (generation.jsonl + run_rag)."""
    model = config.GENERATOR_MODEL
    pr = config.PRICING[model]
    in_tok = out_tok = 0
    for r in rows:
        for c in r.get("llm_calls", []):
            in_tok += c.get("input_tokens") or est_tokens(
                "x" * c.get("prompt_chars", 0)
            )
            out_tok += c.get("output_tokens") or 0
        if not r.get("llm_calls"):
            out_tok += est_tokens(r.get("answer") or "")
    usd = in_tok / 1e6 * pr["input"] + out_tok / 1e6 * pr["output"]
    emb_tokens = sum(est_tokens(r["question"]) for r in rows)
    emb_usd = emb_tokens / 1e6 * config.PRICING[config.EMBEDDING_MODEL]["input"]
    gen_lat = [r["e2e_latency_s"] - r.get("retrieval_latency_s", 0.0) for r in rows]
    e2e = [r["e2e_latency_s"] for r in rows]
    return {
        "n": len(rows),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "embedding_tokens_estimated": emb_tokens,
        "usd_generation_total": usd,
        "usd_embedding_total": emb_usd,
        "usd_total": usd + emb_usd,
        "usd_per_query": (usd + emb_usd) / len(rows) if rows else float("nan"),
        "generation_ms": {f"p{p}": pct(gen_lat, p) * 1000 for p in (50, 95, 99)},
        "e2e_ms": {f"p{p}": pct(e2e, p) * 1000 for p in (50, 95, 99)},
    }


def main() -> None:
    items = config.load_eval_set()
    judged = load(config.RESULTS / "judged.jsonl")
    latency = (
        json.loads((config.RESULTS / "latency_summary.json").read_text())
        if (config.RESULTS / "latency_summary.json").exists()
        else {}
    )
    e2e_sub = (
        json.loads((config.RESULTS / "e2e_latency_summary.json").read_text())
        if (config.RESULTS / "e2e_latency_summary.json").exists()
        else {}
    )
    prod_rows = [
        r
        for r in load(config.RESULTS / "generation.jsonl")
        if r.get("status") == "ok" and r.get("mode") == "legal"
    ]
    gen_rows = load(config.RESULTS / "generators.jsonl")

    systems = {}
    for system in sorted({r["system"] for r in judged}):
        jrows = [r for r in judged if r["system"] == system]
        entry = judge_metrics(jrows)
        entry["by_category"] = {}
        for cat in ("cpc-sections", "case-laws"):
            sub = [r for r in jrows if r.get("category") == cat]
            if sub:
                entry["by_category"][cat] = judge_metrics(sub)
        judged_ids = {r["id"] for r in jrows if r.get("verdict")}
        if system.startswith("production:"):
            entry["cost"] = production_cost([r for r in prod_rows if r["id"] in judged_ids])
        else:
            grows = [
                r
                for r in gen_rows
                if r["system"] == system
                and r["id"] in judged_ids
                and r.get("answer")
                and not r.get("error")
            ]
            entry["cost"] = model_cost(grows, system)
        systems[system] = entry

    summary = {
        "date": date.today().isoformat(),
        "eval_items": len(items),
        "eval_by_category": dict(Counter(it["anchor_category"] for it in items)),
        "systems": systems,
        "production_run": {
            "model": config.GENERATOR_MODEL,
            "rows_ok_legal": len(prod_rows),
            "note": "real src.backend.rag.run_rag on the production pipeline (mode gate + weak-retrieval fallback + temp 0.2)",
        },
        "latency": {"retrieval": latency, "e2e_substitute_generator": e2e_sub},
        "cost": {
            "pricing": config.PRICING,
            "pricing_source": config.PRICING_SOURCE,
        },
    }
    out = config.RESULTS / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
