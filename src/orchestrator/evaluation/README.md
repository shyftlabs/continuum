# TaxPilot RAG Evaluation

## Prerequisites

Docker must be running with the full stack (pgvector + API):

```bash
cd apps/api/shyftlabs-continuum-customer-os
./scripts/docker-rebuild-and-logs.sh build
```

All commands below run from the `src` directory:

```bash
cd apps/api/shyftlabs-continuum-customer-os/src
```

---

## Running the Evaluation

### Option 1 — Full pipeline (regenerate dataset + eval)

Generates ~408 golden cases from pgvector, then scores with RAGAS and DeepEval.

```bash
python -m orchestrator.evaluation.build_golden_dataset --run-eval
```

**Estimated time:** ~3 hours (generation ~40 min, RAGAS ~50 min, DeepEval ~90 min)

### Option 2 — Eval only (reuse existing golden_cases.json)

Skips regeneration and scores the existing dataset.

```bash
python -m orchestrator.evaluation.build_golden_dataset \
  --eval-only orchestrator/evaluation/eval-dataset/golden_cases.json
```

**Estimated time:** ~2 hours

### Option 3 — Generate dataset only (no scoring)

```bash
python -m orchestrator.evaluation.build_golden_dataset
```

---

## Output Files

Saved to `eval-dataset/` (gitignored):

| File | Description |
|---|---|
| `golden_cases.json` | ~408 EvalCases with ground truth Q&A pairs |
| `golden_ragas.json` | RAGAS scores (faithfulness, answer_relevancy, context_precision, context_recall) |
| `golden_deepeval.json` | DeepEval scores (AnswerRelevancy, Faithfulness, TaxAccuracy) |
| `golden_cases_partial.json` | Checkpoint saved every 20 cases during generation (deleted on completion) |

---

## Score Interpretation

| Metric | Evaluator | What it measures |
|---|---|---|
| `faithfulness` | RAGAS | Are all answer statements supported by retrieved chunks? |
| `answer_relevancy` | RAGAS | Does the answer address the question? |
| `context_precision` | RAGAS | Are the most relevant chunks ranked highest? |
| `context_recall` | RAGAS | Did retrieval find the chunk containing the ground truth answer? |
| `Answer Relevancy` | DeepEval | Does the answer address the question? |
| `Faithfulness` | DeepEval | Does the answer contradict the retrieved context? |
| `TaxAccuracy` | DeepEval | Does the answer accurately describe US federal tax law? |

Pass threshold for all metrics: **≥ 0.7**

A case is `overall_passed` only if **all** metrics pass.

---

## Dataset Composition

~408 cases sampled across 7 source types:

| Source | Cases | Description |
|---|---|---|
| `irc` | 100 | IRC sections from Cornell LII |
| `appellate` | 100 | Appellate court opinions (CourtListener) |
| `tax_court` | 50 | Tax Court opinions (CourtListener) |
| `fed_claims` | 50 | Federal Claims Court opinions (CourtListener) |
| `treasury_reg` | 80 | Treasury Regulations (eCFR) |
| `scotus` | 15 | Supreme Court opinions (CourtListener) |
| `treaty` | 13 | US tax treaties (IRS PDFs) |

Each case has a question generated from the full document context and a ground truth answer scoped to a single chunk — enabling meaningful `context_recall` scoring.
