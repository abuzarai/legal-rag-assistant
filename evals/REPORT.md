# RAG Evaluation Report — legal-rag-assistant

**Date:** 2026-10-03 · **Production run completed:** 2026-10-01 · **Frontier baseline completed:** 2026-10-03 · **Evaluator:** agent-run harness in `/evals` · **System:** production
production Weaviate (5,283 chunks) + production retrieval code.

Every number below is produced by a script in `evals/` and stored under
`evals/results/`. Nothing is estimated except where explicitly labelled.

---

## 1. System under test

Full config in [`evals/PIPELINE.md`](PIPELINE.md). In brief:

| Component | Value |
|---|---|
| Index | production Weaviate (1.28), collection `LegalChunk`, **5,283 chunks** (4,904 `cpc-sections` / 379 `case-laws`, 34 source docs) |
| Chunking | `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)` |
| Embedding | `gemini-embedding-001`, 768 dims (matches container env; verified identical vectors across keys) |
| Retrieval | Weaviate hybrid, **alpha=0.5**; fetch `k`; production pools **3 CPC + 2 case-law** chunks (statute-first). The local cosine re-rank was measured and removed — see §3.1 |
| Generator (production) | `gemini-2.5-flash`, temperature 0.2, strict grounded-citation prompt |
| Query gate | `detect_mode()` (social / uncertain / legal) + `is_retrieval_weak()` fallback |

The repo `.env` points at a **stale** standalone Weaviate (545 chunks). The harness
resolves the running production container instead (`config.resolve_weaviate_url`).

---

## 2. Evaluation set construction

| Property | Value |
|---|---|
| Items | **107** (63 `case-laws`, 44 `cpc-sections`) |
| File | [`evals/eval_set.jsonl`](eval_set.jsonl) (committed) |
| Built | 2026-09-14 |
| Question generator | `gemini-3.5-flash-lite`, temp 0, JSON schema |
| Anchors | sampled across all 34 docs: budget split 50/50 by category, then `sqrt(chunk_count)` within a category |
| Question types | procedural 53, holding 27, consequence 14, definitional 12, scenario 1 |
| Relevance labels | anchor chunk + any chunk containing the generator's **verbatim support quote** (≥60 chars, enforced to be a substring of the anchor) |
| Gold set size | mean **1.07** chunks/item; 7/107 items have >1 positive |

**Process.** For each sampled chunk, the generator wrote a natural question, a
reference answer, and a verbatim support quote. `assemble_eval_set.py` dropped
questions that could not be grounded, de-duplicated near-identical questions
(Jaccard ≥ 0.85), and built the gold set by quote containment.

**Manual verification.** Every item was read against its support quote; four
citation fields were corrected after the anchor text showed a different provision
number (q063 → Order XXI R.36; q087 → Order XII R.7; q092 → Order XLI R.25;
q106 → Order XXIII R.3). Four detail-heavy answers were re-read against the full
anchor. Verdicts in [`evals/labels/manual_review.json`](labels/manual_review.json)
(107 accept, 4 with correction). One annotator (the evaluating agent); no
independent second annotator.

---

## 3. Results

### 3.1 Retrieval — chunk-level (primary metric)

`evals/results/retrieval.json` · N=107 · production retrieval code, identical
`gemini-embedding-001` query vectors shared across configs.

| config | R@1 | R@5 | R@10 | P@5 | MRR | NDCG@10 |
|---|---|---|---|---|---|---|
| bm25 | 0.449 | 0.748 | 0.804 | 0.161 | 0.589 | 0.642 |
| dense | 0.579 | 0.771 | 0.818 | 0.159 | 0.673 | 0.700 |
| **hybrid** | 0.575 | **0.869** | **0.944** | **0.183** | **0.721** | **0.771** |
| hybrid+rerank | 0.589 | 0.813 | 0.860 | 0.174 | 0.688 | 0.730 |
| production (statute-first, k=5) | 0.224 | 0.832 | 0.832 | 0.174 | 0.397 | 0.503 |
| production (statute-first, k=10) | 0.224 | 0.364 | 0.925 | 0.073 | 0.360 | 0.489 |
| production k=5 +rerank (removed) | 0.280 | 0.766 | 0.766 | 0.161 | 0.408 | 0.494 |

> **Re-ranker removed.** On the production pooling setup the cosine re-rank
> traded coverage for top-1 precision: dropping it lifts Recall@5 from 0.766 to
> **0.832** (+6.5 pts) and Recall@10 to 0.832, at the cost of R@1 (0.280 → 0.224)
> and MRR (0.408 → 0.397). Since all five retrieved chunks are passed to the
> generator, coverage is what makes an answer possible, so the re-rank was
> removed. It remains a genuine trade-off on CPC alone, where NDCG@10 falls
> 0.745 → 0.720; see §3.2. The whole-corpus `hybrid` row still leads the table.

### 3.2 Retrieval — per category (chunk-level)

| config | cpc R@1 | cpc R@5 | cpc NDCG@10 | case R@1 | case R@5 | case NDCG@10 |
|---|---|---|---|---|---|---|
| bm25 | 0.409 | 0.727 | 0.607 | 0.476 | 0.762 | 0.667 |
| dense | 0.682 | 0.841 | 0.784 | 0.508 | 0.722 | 0.641 |
| hybrid | 0.545 | 0.886 | 0.742 | 0.595 | 0.857 | 0.791 |
| hybrid+rerank | 0.682 | 0.841 | 0.784 | 0.524 | 0.794 | 0.692 |
| production k=5 | 0.545 | 0.841 | 0.720 | **0.000** | 0.825 | 0.352 |
| production k=10 | 0.545 | 0.886 | 0.746 | **0.000** | 0.000 | 0.310 |
| production k=5 +rerank | 0.682 | 0.795 | 0.745 | **0.000** | 0.746 | 0.320 |

### 3.3 Retrieval — page-level and document-level

| config | page R@5 | page R@10 | page NDCG@10 | doc R@5 | doc R@10 | doc NDCG@10 |
|---|---|---|---|---|---|---|
| bm25 | 0.879 | 0.907 | 0.782 | 0.916 | 0.935 | 0.844 |
| dense | 0.879 | 0.897 | 0.812 | 0.897 | 0.907 | 0.848 |
| **hybrid** | **0.963** | **0.972** | **0.877** | **0.981** | **0.991** | **0.914** |
| hybrid+rerank | 0.907 | 0.925 | 0.830 | 0.925 | 0.935 | 0.866 |
| production k=5 | 0.902 | 0.902 | 0.627 | 0.972 | 0.972 | 0.665 |
| production k=5 +rerank | 0.860 | 0.860 | 0.615 | 0.925 | 0.925 | 0.654 |

> Page-level is only meaningful for case-law PDFs; every CPC chunk has
> `page=1` (`.txt` ingestion), so the CPC page metric degenerates to
> document-level. Chunk-level is the honest comparison for CPC.

### 3.4 Generation & grounding

Judge = `gemini-3.1-flash-lite`, temp 0. `evals/results/summary.json`, raw
verdicts `evals/results/judged.jsonl`.

**Production pipeline** (`src.backend.rag.run_rag`: mode gate + weak-retrieval
fallback + temp 0.2, real `gemini-2.5-flash`):

| metric | production `gemini-2.5-flash` (N=107) |
|---|---|
| answer correct | **96.3%** |
| answer partial | 2.8% |
| answer incorrect | 0.9% |
| faithfulness grounded | **98.1%** |
| citation accuracy | **99.1%** |
| hallucination rate | **0.9%** |
| correctness mean (1–5) | 4.91 |
| cost per query | **$0.0017** |
| generation p50 / p95 / p99 | 4,171 / 41,158 / 85,889 ms |
| by category | CPC 41/44 correct · case-law 62/63 |

> **Complete: 107 of 107** (production run finished 2026-10-01). The five
> non-perfect verdicts are mostly answer-relevance failures: q045 incorrect
> (failed to answer), q069, q076 and q097 partially correct (omitted
> reference-answer details). All four were grounded. q077 was judged correct but
> ungrounded, mis-cited and flagged for hallucination.

> **Retrieval path:** these numbers were captured on the current production
> retrieval path (cosine re-ranker removed, see §3.1), so they no longer need
> re-measuring for the re-ranker change.

Substitute generators on the identical retrieved context (production prompt):

| metric | gemini-3.5-flash-lite (N=107) | gemini-3-flash-preview (N=107) |
|---|---|---|
| answer correct | 90.7% | 96.3% |
| answer partial | 7.5% | 0.9% |
| answer incorrect | 1.9% | 2.8% |
| faithfulness grounded | 97.2% | 98.1% |
| citation accuracy | 96.3% | 99.1% |
| hallucination rate | 1.9% | 0.0% |
| correctness mean (1–5) | 4.79 | 4.87 |

By category (substitute generator): `gemini-3.5-flash-lite` — `cpc-sections` 86.4%
correct (38/44), 95.5% grounded; `case-laws` 93.7% correct (59/63), 98.4% grounded.
`gemini-3-flash-preview` — `cpc-sections` 100% correct (44/44), 100% grounded;
`case-laws` 93.7% correct (59/63), 96.8% grounded.

### 3.5 Latency (stated N)

`evals/results/latency_summary.json`, `e2e_latency_summary.json`.

| stage | N | p50 | p95 | p99 |
|---|---|---|---|---|
| Weaviate search only | 100 | 38.9 ms | 56.0 ms | 72.2 ms |
| query embedding | 100 | 698 ms | 2,592 ms | 7,801 ms |
| retrieval total (embed+search) | 100 | 735 ms | 2,635 ms | 7,839 ms |
| end-to-end, flash-lite generator | 30 | 4,554 ms | 9,617 ms | 9,741 ms |
| generation only, flash-lite | 107 | 1,992 ms | 19,219 ms | 40,603 ms |
| end-to-end, production `gemini-2.5-flash` | 107 | 4,301 ms | 41,252 ms | 85,916 ms |
| generation only, production `gemini-2.5-flash` | 107 | 4,171 ms | 41,158 ms | 85,889 ms |
| generation only, 3-flash-preview | 107 | 9,627 ms | 60,807 ms | 207,354 ms |

Embedding p95/p99 are inflated by free-tier throttling (`embed_content` 100 rpm);
the store itself is consistently <125 ms. N is small for p99; treat p99 as indicative.

### 3.6 Cost (list price × measured tokens)

Pricing: `gemini-2.5-flash` $0.30/$2.50, `gemini-3.5-flash-lite` $0.30/$2.50,
`gemini-3.1-flash-lite` $0.25/$1.50, `gemini-3-flash-preview` $0.50/$3.00,
`gemini-embedding-001` $0.15 per 1M input tokens
(source: <https://ai.google.dev/gemini-api/docs/pricing>, retrieved 2026-09-14).

| item | value |
|---|---|
| **production generation, `gemini-2.5-flash`** (107 queries, 189,349 in + 49,359 out) | **$0.0017 / query** |
| flash-lite generation: 189,349 in + 200,545 out tokens (107 queries) | $0.0052 / query |
| embedding: 3,660 estimated tokens (107 queries) | $0.00001 / query |
| 3-flash-preview generation (107 queries, 189,349 in + 255,275 out) | $0.0080 / query |

All keys were on the free tier, so the **actual billed cost of this run was $0**;
the figures above are list-price costs from token counts.

---

## 4. Baseline deltas

Same 107 queries, chunk-level:

- **hybrid vs dense:** Recall@5 0.771 → 0.869 (+9.8 pts); NDCG@10 0.700 → 0.771 (+7.1).
- **hybrid vs BM25:** Recall@5 0.748 → 0.869 (+12.1 pts); NDCG@10 0.642 → 0.771 (+12.9).
- **hybrid+rerank vs hybrid:** Recall@5 0.869 → 0.813 (**−5.6 pts**); NDCG@10 0.771 → 0.730 (−4.1).
  The production cosine re-ranker is a **regression** on this set: it re-scores
  the hybrid candidate pool by the same embedding used for dense retrieval, which
  discards BM25's lexical signal. It does raise R@1 slightly (0.575 → 0.589).
- **production statute-first vs hybrid:** at k=5 the merged list (without the
  removed re-ranker) scores R@5 0.832. Because CPC chunks are always placed
  first, **no case-law question can have a relevant chunk at rank 1**
  (`case-laws R@1 = 0.000`, MRR 0.201) even though 82.5% of case-law items have
  their gold inside the returned 5.

---

## 5. What failed / could not be measured

These are reported as failures, not worked around with fabricated numbers:

1. **Production generator run is complete (107/107).** The per-account daily
   quota for the production model is small (the documented free-tier cap of
   20 requests/day per project per model, shared across the account's keys), so
   the run took several days of resumable
   batches; it finished 2026-10-01. Billing could not be activated, so the cap still
   applies to any re-run.
2. **`gemini-3.5-flash` / `gemini-3.8-flash` / `gemini-3.7-flash`**
   were unusable at various points (20/day cap, or 503). `gemini-2.5-pro` is 404 for
   these keys, so a Pro baseline was impossible.
3. **Pooled (TREC-style) gold expansion was abandoned.** `expand_gold.py` was
   written and run but exhausted the 3.5-flash judge budget; the 107 items keep
   conservative anchor+quote gold (mean 1.07 positives).
4. **Frontier baseline is now complete (N=107/107).** Earlier passes stalled on
   `q035` (repeated `503 UNAVAILABLE`) and on daily-quota `429`s; the resume guard
   retries verdict-less and error rows, so later `run_generators` / `judge` runs
   picked them up without re-doing anything else. The final run finished
   2026-10-03.
5. The production endpoint was not exercised through HTTP; the harness calls the
   production retrieval functions directly. `is_retrieval_weak()` was checked
   post-hoc and triggers on **0/107** items, so the substitute runner bypassing it
   did not change any answer.

---

## 6. Limitations (read before quoting)

- **Circular eval set.** Questions are generated *from* corpus chunks, so the
  anchor is relevant by construction. This favours retrievers that are good at
  the generator's phrasing; Recall is a **lower bound** because ~93% of items
  have a single positive (mean gold 1.07) even where several chunks would answer.
  Some questions name the exact CPC provision, which also helps lexical BM25.
- **Corpus skew.** 93% of chunks are CPC text, only 379 are case law; 107 items
  (63 case-law / 44 CPC) is small, and `CPC_text_part3_of19.txt` is absent from
  the index. `.txt` CPC sources all carry `page=1`, so page-level CPC metrics are
  meaningless.
- **Judge reliability is weakly validated.** On 16 human-labelled answers the
  judge agreed 100% on citation accuracy and hallucination, 93.8% on correctness
  (κ 0.64) and 93.8% on groundedness (κ 0.0 — degenerate, the labels are
  near-uniform). The sample is small and mostly clean, so this mostly shows the
  judge is not over-strict. The judge is a *lite* model, not the production
  model.
- **Substitute vs production generator.** Production `gemini-2.5-flash` is now
  measured on the full **N=107**. The comparison arm (`gemini-3.5-flash-lite`,
  N=107) is a substitute — the prompt and retrieved context are identical to
  production, the model is not. Do not present its 97.2% groundedness figure as
  the production model's score.
- **Single run, single pass.** No variance across seeds/runs; retrieval is
  deterministic, generation is not (production temp 0.2 was not exercised).
- **Latency** was measured from one host; embedding latency is dominated by
  free-tier throttling and is not representative of a paid deployment. The
  `gemini-3-flash-preview` arm is far slower than the others (generation p95
  61 s, p99 207 s), so it is not a practical choice on latency grounds.

---

## 7. Candidate resume bullet

> Built a reproducible RAG evaluation harness (Weaviate v4 hybrid search,
> `gemini-embedding-001`) over a 5,283-chunk Pakistani-law corpus and labelled
> 107 queries: hybrid retrieval reached **Recall@5 0.87 / NDCG@10 0.77** vs
> 0.77 / 0.70 dense-only and 0.75 / 0.64 BM25-only, and showed the production
> cosine re-ranker was a regression (Recall@5 0.81).

Second line, supported by the real production run (full N=107, completed 2026-10-01):

> Production `gemini-2.5-flash` answered with **98.1% groundedness, 99.1% citation
> accuracy and 0.9% hallucination** at **$0.0017 per query** (LLM judge, 16-item
> human check); a 107-query substitute-generator run measured 90.7% correct /
> 97.2% grounded.

---

## 8. Reproduce

```sh
cd legal-rag-assistant   # repo root
uv run python -m evals.dump_corpus            # production index → evals/data/corpus.jsonl
uv run python -m evals.gen_questions           # 140 candidates (free-tier paced)
uv run python -m evals.assemble_eval_set       # → evals/eval_set.jsonl
uv run python -m evals.run_retrieval           # → evals/results/retrieval.json
uv run python -m evals.run_latency             # → evals/results/latency_summary.json
uv run python -m evals.run_generation         # production model through run_rag
uv run python -m evals.run_generators          # substitute generators
uv run python -m evals.run_e2e                 # end-to-end latency
uv run python -m evals.judge                   # LLM judge
uv run python -m evals.validate_judge --score  # judge vs human labels
uv run python -m evals.score                   # → evals/results/summary.json
```

## 9. Traceability

| reported number | script | result file |
|---|---|---|
| Recall@k, P@5, MRR, NDCG@10 | `run_retrieval.py` | `results/retrieval.json`, `results/retrieval_rankings.jsonl` |
| answer correct / faithful / cited / hallucinated | `judge.py` + `score.py` | `results/judged.jsonl`, `results/summary.json` |
| judge agreement | `validate_judge.py` | `results/judge_validation.json`, `labels/judge_human_labels.json` |
| retrieval latency | `run_latency.py` | `results/latency_summary.json`, `results/latency.jsonl` |
| end-to-end latency | `run_e2e.py` | `results/e2e_latency_summary.json` |
| cost | `score.py` (token counts from generators) | `results/summary.json` |
| eval-set provenance | `gen_questions.py` + `assemble_eval_set.py` | `eval_set.jsonl`, `labels/manual_review.json` |
