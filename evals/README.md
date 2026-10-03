# Evaluation Harness — `legal-rag-assistant`

This directory is a **local measuring stick for the RAG service**. It is not part
of the deployed product — the Dockerfile ships only `src/`. The harness exists
because a RAG system is a black box with two moving parts (retrieval + an LLM),
and without a fixed set of questions with known right answers you cannot tell
whether a change made things better or worse. Tuning by intuition ("this answer
felt better") is not reproducible.

It answers three concrete questions:

1. **Does retrieval find the right passages?** — deterministic, cheap, repeatable.
2. **Does generation answer correctly and stay faithful to what was retrieved?** —
   requires a grader.
3. **At what cost and speed?**

The numbers in [`REPORT.md`](REPORT.md) come from this harness. The primary
purpose is engineering: it justified decisions such as "hybrid beats dense",
"the cosine reranker is a regression", and "statute-first pooling costs case-law
rank-1".

---

## 1. The core idea

There was no labelled dataset, so the harness **manufactures one from the corpus
itself**:

> Pick a real chunk from the index → ask an LLM to write a question that chunk
> answers → that chunk becomes the answer key.

Because the question is written *from* the chunk, the chunk is relevant by
construction. This is the origin of the "circular eval set" caveat discussed in
[§7](#7-what-this-does-not-prove).

## 2. The cast of characters

| Term | What it is |
|---|---|
| **Corpus** | The 5,283 chunks in the production Weaviate index (≈93% CPC statute text, ≈7% case-law PDFs) |
| **Chunk** | A ~1,000-character slice of a document (100-char overlap) — the unit the system stores and retrieves |
| **Eval set** | `eval_set.jsonl` — 107 questions, each with a reference answer and a set of "gold" chunks |
| **System under test** | `src.backend.rag.run_rag` — the real production code path |
| **Comparison generators** | Alternative models run on the identical retrieved context and prompt, so differences reflect only the model |
| **Judge** | A separate LLM (`gemini-3.1-flash-lite`) that scores each answer against the reference and the retrieved context |
| **Human labels** | 16 answers a human scored, used to check whether the judge can be trusted |

## 3. How it works

```
 corpus ──► gen_questions ──► candidates.jsonl ──► assemble_eval_set ──► eval_set.jsonl
 (5283)      (LLM writes          (138 candidate        (dedupe, build        (107 items)
             Q + ref + quote)      Q/A/quote pairs)       gold sets)

 eval_set ──► run_retrieval ──► retrieval.json          ← pure retrieval, no LLM
          ──► run_latency   ──► latency_summary.json
          ──► run_generation──► generation.jsonl        ← real run_rag, production model
          ──► run_generators──► generators.jsonl        ← comparison models
          ──► run_e2e       ──► e2e_latency_summary.json
                                    │
                                    ▼
                              judge ──► judged.jsonl    ← LLM grades each answer
                              validate_judge ──► judge_validation.json  ← judge vs human
                              score ──► summary.json    ← one roll-up table
```

- **`gen_questions`** — for each sampled chunk (the "anchor"), an LLM returns a
  question, a reference answer, and a **verbatim support quote** copied from the
  chunk. The quote lets the harness find *other* chunks carrying the same passage.
- **`assemble_eval_set`** — drops unusable candidates, removes near-duplicate
  questions, and builds the **gold set** (answer key) per item: the anchor chunk
  plus any chunk containing the support quote. Assigns ids `q001…q107`.
- **`run_retrieval`** — runs several retrieval strategies over all 107 questions
  and scores each ranking. No LLM involved, so this is deterministic.
- **`run_generation`** — calls the *real* `run_rag`, so the mode gate, the
  weak-retrieval fallback, the prompt, and statute-first pooling are all
  exercised. Records the answer, the retrieved chunks, token usage, and latency.
- **`run_generators`** — replays the same retrieved context and prompt through
  alternative models for an apples-to-apples model comparison.
- **`judge`** — sends each `(question, reference, context, answer)` tuple to the
  judge LLM and records a structured verdict.
- **`score`** — aggregates everything into `summary.json`.

## 4. A real item, traced end to end (`q001`)

```
Question:   "What was the ultimate outcome of the petition as determined by the court?"
Anchor:     case-laws/1991CLC1826.pdf p.5        (the chunk the question was written from)
Gold set:   1 chunk (just the anchor; the support quote added no extras)
Reference:  "The court found the petition to be without any force and accordingly
             dismissed it with no order as to costs."

What retrieval actually returned:
   CPC text part18 of19.txt p.1
   CPC text part14 of19.txt p.1
   1998 M L D 1818.pdf p.2
   1998 M L D 1818.pdf p.11
   (+1)

System answer: "**Summary:** The petition (Petition No.D-1524 of 1997) was dismissed
                due to a lack of merit and being hit by laches, with no order as to
                costs (1998 M L D 1818.pdf p.11)."

Judge:      correct (5/5) · grounded ✓ · citation accurate ✓ · no hallucination
```

Read carefully: **the gold chunk was never retrieved**, yet the answer was judged
correct — because the reference answer is generic ("petition dismissed, no
costs"), and a *different* case in the context supports the same conclusion. Two
lessons:

- Retrieval quality and answer quality are **separate measurements**; look at both.
- The judge grades against the **reference answer**, so a generic reference can
  let an adjacent retrieval score as "correct". That is a property of the eval
  set, not a bug in the harness.

---

## 5. Metrics and terminology

### 5.1 Retrieval metrics (binary relevance)

Retrieval returns a **ranked list** of chunks (rank 1 = best). The **gold set** is
the known-correct chunks for that question. A **hit** is a returned chunk that is
in the gold set.

| Metric | Plain meaning | Formula |
|---|---|---|
| **Recall@k** | Of all the correct chunks, how many did we find in the top *k*? | `(gold ∩ top-k) / |gold|` |
| **Precision@k** | Of the top *k* we returned, how many were actually correct? | `(gold ∩ top-k) / k` |
| **MRR** | How high up is the *first* correct chunk? | `1 / rank_of_first_hit` (0 if none) |
| **NDCG@10** | Like recall, but rewards finding correct chunks *higher* in the list, with diminishing weight | `DCG / ideal DCG`, gain `1/log₂(rank+1)` |

**Toy example.** Truth is `{A, C}`; the system returns `[B, A, D, C, E]`.

| metric | value | why |
|---|---|---|
| Recall@1 | 0/2 = 0.00 | top-1 is B, not relevant |
| Recall@5 | 2/2 = 1.00 | both A and C appear in the top 5 |
| Precision@5 | 2/5 = 0.40 | 2 of the 5 returned are right |
| MRR | 1/2 = 0.50 | first hit (A) is at rank 2 |
| NDCG@10 | < 1.0 | correct items are at ranks 2 and 4, not 1 and 2 |

**Recall@5 vs NDCG@10** are the two headline retrieval numbers:

- Recall@5 asks *"is the answer in the top 5?"* — a **coverage** question.
- NDCG@10 asks *"and is it near the top?"* — a **ranking-quality** question.

They can disagree. Example: `production@10` reaches Recall@10 = 0.925 (good
coverage) but NDCG@10 = 0.489 (weak ordering), because statute-first pooling
pushes CPC chunks into the top ranks.

**Aggregate** simply means the **mean across all 107 questions**.

### 5.2 Retrieval strategies (the ablation)

An **ablation** tests variants of one component while holding everything else
fixed, to see what each piece contributes.

| Strategy | What it does |
|---|---|
| **BM25** | Classic keyword matching (lexical overlap). Strong when the question uses the exact words in the text. |
| **Dense** | Pure embedding similarity (`near_vector`) — matches *meaning*, not words. |
| **Hybrid** | Weaviate fuses BM25 + vector. `alpha=0.5` weights them equally (1.0 = pure vector, 0.0 = pure BM25). |
| **Hybrid + rerank** | Fetch a wide pool (`max(k×3, 10)`), then re-sort by cosine similarity between the query vector and each chunk's stored vector. |
| **Production** | The real pipeline: 3 chunks from `cpc-sections` + 2 from `case-laws`, merged **statute-first**. |
| **Production@10** | Same, but 6 + 4 (the `k=10` variant). |

Supporting terms:

- **Embedding / vector / dims** — a chunk is converted to a list of 768 numbers
  that encodes its meaning. Similar meaning → similar vector.
- **Cosine similarity** — how aligned two vectors are (1 = identical direction).
  **Cosine distance** = `1 − similarity`, so *lower is better*. Weaviate returns
  `distance` for vector search and `score` for BM25.
- **Reranker** — the local cosine re-sort between query and chunk vectors. It is
  measured here as an ablation but was removed from production (REPORT §3.1).
- **Statute-first pooling** — always placing CPC statutory text ahead of case law.
  That is why **case-law Recall@1 is 0.00 in production**: a case-law chunk
  physically cannot occupy rank 1.

### 5.3 Generation metrics (LLM-as-judge)

There is no single "correct string" for an answer, so exact matching fails.
Instead, a judge LLM reads `(question, reference answer, retrieved context,
system answer)` and scores four dimensions:

| Dimension | Question it answers | Values |
|---|---|---|
| **Answer correctness** | Does the answer agree with the reference? | `correct` / `partially_correct` / `incorrect`, score 1–5 |
| **Faithfulness (groundedness)** | Is *every* claim supported by the retrieved context? | `grounded` true/false + list of `unsupported_claims` |
| **Citation accuracy** | Are the cited provisions/cases actually in the context and attached to the right sentence? | `accurate` true/false + `invalid_citations` |
| **Hallucination** | Did it invent a case, section, number, date, or fact? | `present` true/false |

**Why groundedness is separate from correctness.** An answer can be legally right
but *not supported by what retrieval returned* — the model supplied it from
memory. In a RAG system that grounding is the whole point, so it is measured on
its own. The judge is explicitly told the context is the *only* admissible
evidence.

**Cohen's κ (kappa)** — a chance-corrected agreement score between the judge and
the human labels. `1.0` = perfect; `0` = no better than guessing. The harness
reports both raw agreement and κ in `judge_validation.json` (n=16): agreement is
100% on citation accuracy and hallucination, and 93.8% on correctness (κ 0.64)
and groundedness (κ 0.0 — degenerate, the labels are near-uniform). The sample
is small, so this mainly shows the judge is not over-strict.

### 5.4 Eval-set vocabulary

| Term | Meaning |
|---|---|
| **Anchor chunk** | The corpus chunk a question was generated from — the primary answer key. |
| **Support quote** | A verbatim substring of the anchor, used to find *other* chunks containing the same passage. |
| **Gold set** | The full answer key: anchor + every chunk containing the support quote. |
| **Multi-gold** | Items with more than one correct chunk (7/107). The rest have exactly one. |
| **Question types** | procedural (53), holding (27), consequence (14), definitional (12), scenario (1). |
| **Leaky item** | A question whose wording overlaps its own source passage by >60% tokens — trivially easy to retrieve. Detected (1 item), not removed. |
| **Circular eval set** | Questions are derived from corpus chunks, so the answer is relevant by construction — this inflates scores relative to real user traffic. |

### 5.5 Operations vocabulary

| Term | Meaning |
|---|---|
| **p50 / p95 / p99** | Percentile latencies. p95 = "95% of requests were faster than this". The tail matters more than the average for user experience. |
| **List-price cost** | Tokens measured in the run × the published per-million-token price. It is a modelled cost, not a bill. |
| **Production vs comparison** | Production = the deployed model through `run_rag`. Comparison = the same prompt and context through another model. |
| **n** | How many items were actually scored for a given arm. Always check it before quoting a rate. |

---

## 6. How to read the results

- **`retrieval.json`** — the ablation table. Compare rows to see what each
  retrieval piece buys. Look for the best Recall@5 / NDCG@10 (hybrid), and for
  any added component that makes things *worse* (the reranker does).
- **`summary.json`** — one block per system with `n`, the four generation rates,
  cost, and latency.
- **`REPORT.md`** — the narrative and limitations. Read §5–6 before quoting any
  number.

Current headline for the production pipeline: **96.3% correct, 98.1% grounded,
99.1% citation-accurate, 0.9% hallucination at ≈$0.0017/query** over 107 items —
subject to the caveats below.

## 7. Reproducing

```sh
uv run python -m evals.dump_corpus            # production index → evals/data/corpus.jsonl
uv run python -m evals.gen_questions          # candidates (LLM-paced)
uv run python -m evals.assemble_eval_set      # → evals/eval_set.jsonl
uv run python -m evals.run_retrieval          # → evals/results/retrieval.json
uv run python -m evals.run_latency            # → evals/results/latency_summary.json
uv run python -m evals.run_generation         # production model through run_rag
uv run python -m evals.run_generators         # comparison models
uv run python -m evals.run_e2e                # end-to-end latency
uv run python -m evals.judge                  # LLM grades each answer
uv run python -m evals.validate_judge --score # judge vs human labels
uv run python -m evals.score                  # → evals/results/summary.json
```

The generation, judging, and question-generation stages are **resumable**: a
re-run skips work that already completed and continues where it stopped.

## 8. What this does not prove

- **It is not real user traffic.** Questions written from chunks are easier than
  questions real lawyers ask, and they favour the wording of the source text.
  Recall is a **lower bound**, and correctness is likely an **over**estimate.
- **One annotator, one run.** There is no inter-annotator agreement on the eval
  set itself, and no seed/run variance (retrieval is deterministic, generation
  is not).
- **Small, skewed set.** 107 items; the corpus is ≈93% CPC by chunk count. `.txt`
  CPC sources all carry `page=1`, so page-level CPC metrics are meaningless.
- **The judge is a lite model, not the production model**, and its validation
  sample is small (16 human-labelled items, with 15/16 agreement on correctness
  and groundedness), so it mainly proves the judge is not over-strict.
- **Latency was measured from a single host** and is network-bound, so it is not
  necessarily representative of a production deployment.
- **The "production" retrieval config is a reimplementation** of the retrieval
  path rather than a literal function call, so it can drift from `src/backend/deps.py`.
