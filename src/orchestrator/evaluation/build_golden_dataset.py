"""
Golden dataset builder for TaxPilot RAG evaluation.

Approach:
  1. Sample N distinct documents (source_ref) per source type from pgvector
  2. Fetch ALL chunks for each document and concatenate into full document text
  3. Send full document context to GPT-4o-mini → generate one question whose
     answer is FULLY contained within a single chunk (chunk-scoped answer)
  4. Call TaxPilot RAG API with the question → get retrieved answer + citations
  5. Build EvalCase with real ground truth (expected_output from chunk text)

This produces ~410 evaluation cases where:
  - expected_output  = precise answer grounded in actual document text
  - context          = what RAG actually retrieved
  - ground_truth_chunk_id = which chunk contains the answer (for retrieval audit)

Usage:
    python build_golden_dataset.py
    python build_golden_dataset.py --run-eval
    python build_golden_dataset.py --output golden_cases.json

Outputs:
    eval-dataset/golden_cases.json       — ~410 EvalCases with real ground truth
    eval-dataset/golden_ragas.json       — RAGAS scores (if --run-eval)
    eval-dataset/golden_deepeval.json    — DeepEval scores (if --run-eval)
"""

from __future__ import annotations

import os

os.environ.setdefault("SESSION_ENABLED", "false")
os.environ.setdefault("MEMORY_ENABLED", "false")

import argparse
import asyncio
import csv
import io
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

from orchestrator.evaluation.types import EvalCase, EvalResult

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).parent / "eval-dataset"
PARTIAL_SAVE_PATH = OUTPUT_DIR / "golden_cases_partial.json"

RAG_API_URL = os.environ.get("TAXPILOT_API_URL", "http://localhost:8000")
RAG_ENDPOINT = f"{RAG_API_URL}/api/v1/research/query"
RAG_TIMEOUT = 60
RAG_DELAY_SECONDS = 1.5

API_TOKEN = os.environ.get("TAXPILOT_API_TOKEN", "")
RANDOM_SEED = 42

# Max total chars of document context sent to GPT-4o-mini for Q&A generation.
# Caps very long docs (SCOTUS opinions can be 400+ chunks).
MAX_DOC_CHARS = 8000
MAX_CHUNK_CHARS = 2000   # per individual chunk
MAX_CHUNKS_PER_DOC = 6   # max chunks to include from a document

# Sampling targets per source type → total ~408 cases
SAMPLE_TARGETS: dict[str, int] = {
    "irc":          100,
    "appellate":    100,
    "tax_court":     50,
    "fed_claims":    50,
    "treasury_reg":  80,
    "scotus":        15,   # all available
    "treaty":        13,   # all available
}


# ---------------------------------------------------------------------------
# pgvector queries via docker exec
# ---------------------------------------------------------------------------

def _psql(sql: str, timeout: int = 60) -> list[dict[str, str]]:
    """Run SQL via docker exec taxpilot_postgres, return list of row dicts."""
    result = subprocess.run(
        ["docker", "exec", "taxpilot_postgres", "psql",
         "-U", "taxpilot", "-d", "taxpilot", "--csv", "-c", sql],
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"psql error: {result.stderr.strip()}")
    reader = csv.DictReader(io.StringIO(result.stdout))
    return list(reader)


def get_source_refs(source_type: str, limit: int) -> list[str]:
    """Return randomly sampled distinct source_refs for a source type."""
    rows = _psql(
        f"SELECT DISTINCT source_ref FROM tax_law_chunks "
        f"WHERE source_type = '{source_type}' ORDER BY source_ref;"
    )
    all_refs = [r["source_ref"] for r in rows]
    rng = random.Random(RANDOM_SEED)
    return rng.sample(all_refs, min(limit, len(all_refs)))


def get_chunks_for_refs(source_type: str, source_refs: list[str]) -> dict[str, list[dict]]:
    """
    Fetch all chunks for the given source_refs.
    Returns dict: source_ref → list of chunk dicts sorted by chunk_index.
    """
    if not source_refs:
        return {}

    escaped = [ref.replace("'", "''") for ref in source_refs]
    in_clause = ", ".join(f"'{r}'" for r in escaped)

    rows = _psql(
        f"SELECT id, source_ref, section_title, chunk_index, "
        f"left(content, {MAX_CHUNK_CHARS}) AS content "
        f"FROM tax_law_chunks "
        f"WHERE source_type = '{source_type}' "
        f"AND source_ref IN ({in_clause}) "
        f"ORDER BY source_ref, chunk_index::int;",
        timeout=120,
    )

    docs: dict[str, list[dict]] = {}
    for row in rows:
        ref = row["source_ref"]
        docs.setdefault(ref, []).append(row)
    return docs


# ---------------------------------------------------------------------------
# Q&A generation from full document context
# ---------------------------------------------------------------------------

_openai_client: OpenAI | None = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def build_doc_context(chunks: list[dict]) -> tuple[str, list[dict]]:
    """
    Concatenate chunks into a single document context string.
    Caps at MAX_CHUNKS_PER_DOC chunks and MAX_DOC_CHARS total.
    Spreads selection evenly across the document so later sections are covered.
    Returns (context_string, included_chunks).
    """
    if len(chunks) > MAX_CHUNKS_PER_DOC:
        sampled = random.sample(chunks, MAX_CHUNKS_PER_DOC)
        included = sorted(sampled, key=lambda c: int(c["chunk_index"]))
    else:
        included = chunks
    parts = []
    total = 0
    for chunk in included:
        content = chunk["content"].strip()
        label = f"[CHUNK {chunk['chunk_index']}]\n{content}"
        if total + len(label) > MAX_DOC_CHARS:
            break
        parts.append(label)
        total += len(label)
    return "\n\n".join(parts), included[:len(parts)]


def generate_qa_from_document(
    source_type: str,
    source_ref: str,
    section_title: str,
    chunks: list[dict],
) -> dict[str, Any] | None:
    """
    Call GPT-4o-mini to generate one (question, answer, chunk_index) triple
    from the full document context. Answer must be contained in a single chunk.

    Returns dict with keys: question, answer, answer_chunk_index, answer_chunk_id
    or None if generation failed.
    """
    doc_context, included_chunks = build_doc_context(chunks)

    if not doc_context.strip():
        return None

    # Source-type specific instruction
    source_hints = {
        "irc": "tax law rules, thresholds, definitions, or eligibility requirements",
        "treasury_reg": "regulatory requirements, compliance rules, or defined terms",
        "tax_court": "the court's holding, the tax issue decided, or the legal reasoning",
        "appellate": "the court's ruling, the legal standard applied, or the outcome",
        "scotus": "the Supreme Court's holding or the legal principle established",
        "fed_claims": "the court's decision or the legal basis for the ruling",
        "treaty": "withholding rates, residency rules, or double taxation relief provisions",
    }
    hint = source_hints.get(source_type, "the key legal or factual content")

    prompt = f"""You are building an evaluation dataset for a tax law RAG (retrieval-augmented generation) system.

Below are text chunks from a single document: "{section_title}" (source: {source_type}).

Your task: generate ONE specific factual question about {hint} in this document.

Rules:
1. The answer must be FULLY contained within a SINGLE chunk shown below — do not combine information across chunks.
2. The answer must be specific: a number, threshold, name, legal standard, or explicit rule — NOT generic.
3. The question should reflect the document topic but be answerable from one chunk.
4. Do not ask about things not present in the text.

Document chunks:
{doc_context}

Respond ONLY with valid JSON (no markdown, no explanation):
{{
  "question": "...",
  "answer": "...",
  "answer_chunk_index": <integer — the CHUNK index number whose text contains the answer>
}}"""

    try:
        client = _get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"},
            timeout=30,
        )
        raw = response.choices[0].message.content
        data = json.loads(raw)

        question = data.get("question", "").strip()
        answer = data.get("answer", "").strip()
        answer_chunk_index = int(data.get("answer_chunk_index", 0))

        if not question or not answer:
            return None

        # Find the chunk with the matching index
        answer_chunk = next(
            (c for c in included_chunks if str(c["chunk_index"]) == str(answer_chunk_index)),
            included_chunks[0],  # fallback to first chunk
        )

        return {
            "question": question,
            "answer": answer,
            "answer_chunk_index": answer_chunk_index,
            "answer_chunk_id": answer_chunk["id"],
            "answer_chunk_content": answer_chunk["content"],
        }

    except Exception as e:
        print(f"    [WARN] GPT-4o-mini failed for '{source_ref}': {e}")
        return None


# ---------------------------------------------------------------------------
# RAG API call
# ---------------------------------------------------------------------------

async def call_rag_api(question: str, client: httpx.AsyncClient) -> dict[str, Any]:
    payload = {"question": question, "force_research": True}
    headers = {}
    if API_TOKEN:
        headers["Authorization"] = f"Bearer {API_TOKEN}"
    try:
        async with client.stream(
            "POST", RAG_ENDPOINT, json=payload, headers=headers, timeout=RAG_TIMEOUT
        ) as resp:
            resp.raise_for_status()
            answer_tokens: list[str] = []
            citations: list[dict] = []

            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[len("data:"):].strip()
                if not raw:
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if event.get("type") == "token":
                    answer_tokens.append(event.get("content", ""))
                elif event.get("type") == "done":
                    citations = event.get("citations", [])

            return {"answer": "".join(answer_tokens), "citations": citations}

    except Exception as exc:
        print(f"    [WARN] RAG API error for '{question[:60]}': {exc}")
        return {"answer": "", "citations": []}


def _citations_to_context(citations: list[dict]) -> list[str]:
    chunks = []
    for c in citations:
        ref = c.get("source_ref", "")
        section_title = c.get("section_title", "")
        content = c.get("content", "") or ""
        header = f"[{ref}] {section_title}".strip()
        chunks.append(f"{header}\n{content}".strip())
    return [ch for ch in chunks if ch]


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_cases(cases: list[EvalCase], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([c.to_dict() for c in cases], indent=2, ensure_ascii=False))
    print(f"Saved {len(cases)} EvalCases → {path}")


def save_results(results: list[EvalResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.to_dict() for r in results], indent=2, ensure_ascii=False))
    print(f"Saved {len(results)} EvalResults → {path}")


def print_summary(results: list[EvalResult], label: str) -> None:
    if not results:
        print(f"\n{label}: no results.")
        return
    passed = sum(1 for r in results if r.overall_passed)
    avg = sum(r.overall_score or 0.0 for r in results) / len(results)
    print(f"\n{label} Summary:")
    print(f"  Cases evaluated  : {len(results)}")
    print(f"  Passed           : {passed}/{len(results)}")
    print(f"  Avg overall score: {avg:.3f}")

    source_scores: dict[str, list[float]] = {}
    for r in results:
        st = r.metadata.get("source_type", "unknown")
        source_scores.setdefault(st, []).append(r.overall_score or 0.0)
    print(f"\n  Per-source breakdown:")
    for st, scores in sorted(source_scores.items()):
        print(f"    {st:<20} avg={sum(scores)/len(scores):.3f}  n={len(scores)}")

    criterion_scores: dict[str, list[float]] = {}
    for r in results:
        for s in r.scores:
            criterion_scores.setdefault(s.criterion, []).append(s.score)
    print(f"\n  Per-criterion breakdown:")
    for criterion, scores in criterion_scores.items():
        print(f"    {criterion:<30} avg={sum(scores)/len(scores):.3f}")


# ---------------------------------------------------------------------------
# RAGAS evaluation
# ---------------------------------------------------------------------------

async def run_ragas(cases: list[EvalCase]) -> list[EvalResult]:
    from orchestrator.evaluation import RagasEvaluator

    evaluator = RagasEvaluator(
        metric_names=["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
    )
    results = []
    for i, case in enumerate(cases):
        if not case.context:
            print(f"  [RAGAS] Skipping case {i+1} — no context retrieved")
            continue
        answer = case.metadata.get("answer", "")
        if not answer:
            print(f"  [RAGAS] Skipping case {i+1} — empty answer")
            continue
        src = case.metadata.get("source_type", "")
        print(f"  [RAGAS] [{src}] Evaluating case {i+1}/{len(cases)}: {case.input_text[:55]}...")
        result = await evaluator.evaluate(case, answer)
        result.metadata["case_id"] = case.case_id
        result.metadata["source_type"] = src
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# DeepEval evaluation
# ---------------------------------------------------------------------------

async def run_deepeval(cases: list[EvalCase]) -> list[EvalResult]:
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
    from deepeval.test_case import LLMTestCaseParams
    from orchestrator.evaluation import DeepEvalEvaluator

    evaluator = DeepEvalEvaluator(
        metrics=[
            AnswerRelevancyMetric(threshold=0.7),
            FaithfulnessMetric(threshold=0.7),
            GEval(
                name="TaxAccuracy",
                criteria=(
                    "The response should accurately describe US federal tax law "
                    "concepts, cite relevant IRC sections or case law, and avoid "
                    "fabricating rules, thresholds, or holdings."
                ),
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                ],
                threshold=0.7,
            ),
        ]
    )
    results = []
    for i, case in enumerate(cases):
        answer = case.metadata.get("answer", "")
        if not answer:
            print(f"  [DeepEval] Skipping case {i+1} — empty answer")
            continue
        src = case.metadata.get("source_type", "")
        print(f"  [DeepEval] [{src}] Evaluating case {i+1}/{len(cases)}: {case.input_text[:55]}...")
        result = await evaluator.evaluate(case, answer)
        result.metadata["case_id"] = case.case_id
        result.metadata["source_type"] = src
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(output_path: Path, run_eval: bool) -> None:
    print("=== TaxPilot Golden Dataset Builder ===\n")
    print("Targets per source type:")
    for st, n in SAMPLE_TARGETS.items():
        print(f"  {st:<15} {n}")
    print(f"  {'TOTAL':<15} {sum(SAMPLE_TARGETS.values())}\n")

    cases: list[EvalCase] = []
    case_num = 0

    for source_type, target in SAMPLE_TARGETS.items():

        print(f"\n── {source_type.upper()} (target: {target}) ──")

        # Step 1: sample source_refs
        print(f"  Sampling {target} documents...")
        source_refs = get_source_refs(source_type, target)
        print(f"  Sampled {len(source_refs)} source_refs")

        # Step 2: fetch all chunks for sampled docs
        print(f"  Fetching chunks from pgvector...")
        docs = get_chunks_for_refs(source_type, source_refs)
        print(f"  Fetched {sum(len(v) for v in docs.values())} chunks across {len(docs)} docs")

        # Step 3 + 4: generate Q&A + call RAG API
        for i, source_ref in enumerate(source_refs):
            chunks = docs.get(source_ref, [])
            if not chunks:
                print(f"    [{i+1}/{len(source_refs)}] SKIP — no chunks for '{source_ref}'")
                continue

            section_title = chunks[0].get("section_title", source_ref)
            case_num += 1
            print(f"    [{i+1}/{len(source_refs)}] {source_ref[:60]} ({len(chunks)} chunks)...")

            # Generate Q&A from full document context
            qa = generate_qa_from_document(source_type, source_ref, section_title, chunks)
            if not qa:
                print(f"      SKIP — Q&A generation failed")
                case_num -= 1
                continue

            print(f"      Q: {qa['question'][:70]}...")
            print(f"      A: {qa['answer'][:70]}...")

            # Call RAG API
            async with httpx.AsyncClient() as client:
                t0 = time.monotonic()
                rag_result = await call_rag_api(qa["question"], client)
                elapsed = time.monotonic() - t0

            if case_num < sum(SAMPLE_TARGETS.values()):
                await asyncio.sleep(RAG_DELAY_SECONDS)

            rag_answer = rag_result["answer"]
            context = _citations_to_context(rag_result["citations"])
            retrieval_hit = any(
                c.get("chunk_id") == qa["answer_chunk_id"]
                for c in rag_result["citations"]
            )

            case = EvalCase(
                input_text=qa["question"],
                expected_output=qa["answer"],          # real ground truth from chunk
                context=context,                        # what RAG actually retrieved
                metadata={
                    "source_type": source_type,
                    "source_ref": source_ref,
                    "section_title": section_title,
                    "ground_truth_chunk_id": qa["answer_chunk_id"],
                    "ground_truth_chunk_index": qa["answer_chunk_index"],
                    "ground_truth_context": qa["answer_chunk_content"],
                    "retrieval_hit": retrieval_hit,
                    "rag_latency_ms": int(elapsed * 1000),
                    "has_context": bool(context),
                    "answer": rag_answer,
                },
            )
            cases.append(case)

            # Save partial progress every 20 cases
            if len(cases) % 20 == 0:
                save_cases(cases, PARTIAL_SAVE_PATH)
                print(f"      [checkpoint] {len(cases)} cases saved so far")

    print(f"\n{'─'*50}")
    print(f"Generated {len(cases)} golden EvalCases total.")

    from collections import Counter
    counts = Counter(c.metadata.get("source_type") for c in cases)
    for st, n in sorted(counts.items()):
        print(f"  {st:<20} {n} cases")

    # Save final dataset
    save_cases(cases, output_path)

    # Clean up partial file
    if PARTIAL_SAVE_PATH.exists():
        PARTIAL_SAVE_PATH.unlink()

    if not run_eval:
        print("\nDone. Use --run-eval to run RAGAS and DeepEval scoring.")
        return

    # RAGAS
    print("\nRunning RAGAS evaluation...")
    try:
        ragas_results = await run_ragas(cases)
        print_summary(ragas_results, "RAGAS")
        save_results(ragas_results, output_path.parent / "golden_ragas.json")
    except ImportError as e:
        print(f"[SKIP] RAGAS not installed: {e}")

    # DeepEval
    print("\nRunning DeepEval evaluation...")
    try:
        deepeval_results = await run_deepeval(cases)
        print_summary(deepeval_results, "DeepEval")
        save_results(deepeval_results, output_path.parent / "golden_deepeval.json")
    except ImportError as e:
        print(f"[SKIP] DeepEval not installed: {e}")

    print("\n=== Done ===")


async def eval_only(cases_path: Path) -> None:
    """Load existing golden_cases.json and run RAGAS + DeepEval without regenerating."""
    print(f"=== TaxPilot Golden Dataset Evaluator ===\n")
    print(f"Loading cases from {cases_path}...")
    raw = json.loads(cases_path.read_text())
    cases = [EvalCase.from_dict(c) for c in raw]
    print(f"Loaded {len(cases)} EvalCases.\n")

    # RAGAS
    print("Running RAGAS evaluation...")
    try:
        ragas_results = await run_ragas(cases)
        print_summary(ragas_results, "RAGAS")
        save_results(ragas_results, cases_path.parent / "golden_ragas.json")
    except ImportError as e:
        print(f"[SKIP] RAGAS not installed: {e}")

    # DeepEval
    print("\nRunning DeepEval evaluation...")
    try:
        deepeval_results = await run_deepeval(cases)
        print_summary(deepeval_results, "DeepEval")
        save_results(deepeval_results, cases_path.parent / "golden_deepeval.json")
    except ImportError as e:
        print(f"[SKIP] DeepEval not installed: {e}")

    print("\n=== Done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build TaxPilot golden evaluation dataset")
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_DIR / "golden_cases.json"),
        help="Output path for golden EvalCase JSON",
    )
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="Run RAGAS and DeepEval after generating the dataset",
    )
    parser.add_argument(
        "--eval-only",
        type=str,
        metavar="CASES_JSON",
        help="Skip generation; load existing cases JSON and run RAGAS + DeepEval only",
    )
    args = parser.parse_args()

    if args.eval_only:
        asyncio.run(eval_only(Path(args.eval_only)))
    else:
        asyncio.run(main(output_path=Path(args.output), run_eval=args.run_eval))
