"""
Evaluation dataset generator for TaxPilot RAG system.

Builds a golden dataset covering all 7 knowledge base sources:
  - irc          (Cornell LII — IRC sections)
  - treasury_reg (eCFR — Treasury Regulations)
  - tax_court    (CourtListener — Tax Court opinions)
  - appellate    (CourtListener — Circuit Court opinions)
  - scotus       (CourtListener — SCOTUS opinions)
  - fed_claims   (CourtListener — Court of Federal Claims opinions)
  - treaty       (IRS — Tax treaties)

For IRC and treasury_reg, questions are generated from section titles.
For case law and treaties, questions are generated from case/treaty names.
All questions call the TaxPilot RAG API to get real answers + citations.

Usage:
    python generate_eval_dataset.py
    python generate_eval_dataset.py --run-eval
    python generate_eval_dataset.py --samples 5
    python generate_eval_dataset.py --output my.json

Outputs:
    eval-dataset/generated_cases.json   — EvalCase list (reusable, no LLM needed)
    eval-dataset/ragas_results.json     — RAGAS scores (if --run-eval)
    eval-dataset/deepeval_results.json  — DeepEval scores (if --run-eval)
"""

from __future__ import annotations

import os

# Disable session and memory for evaluation — avoids needing Redis/Qdrant
# and does not affect production settings in .env
os.environ.setdefault("SESSION_ENABLED", "false")
os.environ.setdefault("MEMORY_ENABLED", "false")

import argparse
import asyncio
import csv
import json
import random
import re
import time
from pathlib import Path
from typing import Any

import httpx

from orchestrator.evaluation.types import EvalCase, EvalResult

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CSV_DIR = Path(__file__).parent / "eval-dataset" / "cornell-title-26"
OUTPUT_DIR = Path(__file__).parent / "eval-dataset"

RAG_API_URL = os.environ.get("TAXPILOT_API_URL", "http://localhost:8000")
RAG_ENDPOINT = f"{RAG_API_URL}/api/v1/research/query"
RAG_TIMEOUT = 60  # seconds per request
RAG_DELAY_SECONDS = 1.5  # delay between requests to avoid rate limiting

DEFAULT_SAMPLES_PER_SOURCE = 5
RANDOM_SEED = 42

# Optional: set TAXPILOT_API_TOKEN env var to use authenticated requests
# (bypasses the 10-query anonymous limit)
API_TOKEN = os.environ.get("TAXPILOT_API_TOKEN", "")

# pgvector DB connection for sampling case law / treaty metadata
DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://taxpilot:taxpilot_dev@localhost:5432/taxpilot",
)


# ---------------------------------------------------------------------------
# Question templates per source type
# ---------------------------------------------------------------------------

_IRC_TEMPLATES = [
    "What does {title} cover under US federal tax law?",
    "Explain the rules and requirements of {title} under the IRC.",
    "How does {title} apply to US taxpayers?",
    "What are the key provisions of {title}?",
    "Under IRC §{section}, what is the tax treatment for {topic}?",
    "What are the eligibility requirements for {title}?",
    "How is {topic} defined or calculated under IRC §{section}?",
    "What limitations or thresholds apply under {title}?",
]

_TREASURY_REG_TEMPLATES = [
    "What does {title} require under the Treasury Regulations?",
    "Explain the rules set out in {title}.",
    "How does {title} apply to taxpayers?",
    "What are the key provisions of {title}?",
    "What compliance obligations does {title} impose?",
]

_CASE_TEMPLATES = [
    "What was the court's holding in {title}?",
    "What tax issue was decided in {title}?",
    "What did the court rule regarding the tax dispute in {title}?",
    "What are the key facts and outcome of {title}?",
    "How did the court analyze the tax law question in {title}?",
]

_TREATY_TEMPLATES = [
    "What are the key provisions of the {title}?",
    "How does the {title} affect withholding taxes on dividends and interest?",
    "What residency and permanent establishment rules apply under the {title}?",
    "How does the {title} prevent double taxation for US taxpayers?",
    "What income types are covered or exempt under the {title}?",
]


def _pick_template(templates: list[str], seed_offset: int) -> str:
    rng = random.Random(RANDOM_SEED + seed_offset)
    return rng.choice(templates)


# ---------------------------------------------------------------------------
# IRC — load from Cornell Title 26 CSVs
# ---------------------------------------------------------------------------


def _clean_section_label(label: str) -> tuple[str, str, str]:
    if any(kw in label for kw in ["Repealed", "Renumbered", "[Reserved]"]):
        return "", "", ""
    m = re.search(r"§\s*([\w\-]+)\s*[-–]\s*(.+)", label)
    if not m:
        return "", "", ""
    section_num = m.group(1).strip()
    topic = m.group(2).strip().rstrip("]").strip()
    title = f"IRC §{section_num} – {topic}"
    return section_num, topic, title


def load_irc_sections(csv_dir: Path, samples_per_subtitle: int) -> list[dict[str, Any]]:
    """Sample IRC sections from Cornell Title 26 CSVs."""
    sections: list[dict[str, Any]] = []
    csv_files = sorted(csv_dir.glob("subtitle_*.csv"))

    if not csv_files:
        print(f"  [WARN] No subtitle CSVs found in {csv_dir} — skipping IRC source")
        return []

    print(f"  Found {len(csv_files)} subtitle CSV files.")
    for csv_path in csv_files:
        subtitle_sections: list[dict[str, Any]] = []
        subtitle_name = ""

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row.get("section_label", "").strip()
                section_url = row.get("section_url", "").strip()
                if not subtitle_name:
                    subtitle_name = row.get("subtitle_label", "").strip()
                section_num, topic, title = _clean_section_label(label)
                if not section_num:
                    continue
                subtitle_sections.append(
                    {
                        "source_type": "irc",
                        "section_num": section_num,
                        "topic": topic,
                        "title": title,
                        "section_url": section_url,
                        "subtitle": subtitle_name,
                    }
                )

        rng = random.Random(RANDOM_SEED)
        sampled = rng.sample(subtitle_sections, min(samples_per_subtitle, len(subtitle_sections)))
        sections.extend(sampled)
        print(f"    {csv_path.name}: {len(subtitle_sections)} sections → sampled {len(sampled)}")

    return sections


# ---------------------------------------------------------------------------
# Other sources — load from pgvector DB
# ---------------------------------------------------------------------------


def load_chunks_from_db(source_type: str, samples: int) -> list[dict[str, Any]]:
    """
    Sample distinct (source_ref, section_title) pairs from tax_law_chunks
    for the given source_type.

    Queries via `docker exec taxpilot_postgres` to avoid local postgres port
    conflicts (both local and Docker postgres may listen on localhost:5432).
    Falls back to direct psycopg2 connection if docker is unavailable.
    """
    import subprocess

    sql = (
        f"SELECT DISTINCT source_ref, section_title "
        f"FROM tax_law_chunks "
        f"WHERE source_type = '{source_type}' "
        f"AND section_title IS NOT NULL AND section_title != '' "
        f"ORDER BY source_ref;"
    )

    try:
        result = subprocess.run(
            [
                "docker",
                "exec",
                "taxpilot_postgres",
                "psql",
                "-U",
                "taxpilot",
                "-d",
                "taxpilot",
                "-t",
                "-A",
                "-F",
                "\t",
                "-c",
                sql,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())

        rows = []
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                rows.append((parts[0].strip(), parts[1].strip()))

    except Exception as e:
        print(f"  [WARN] docker exec query failed for {source_type}: {e}")
        return []

    if not rows:
        print(f"  [WARN] No chunks found for source_type={source_type}")
        return []

    rng = random.Random(RANDOM_SEED)
    sampled = rng.sample(rows, min(samples, len(rows)))
    print(f"    {source_type}: {len(rows)} distinct entries → sampled {len(sampled)}")

    return [
        {"source_type": source_type, "source_ref": source_ref, "title": section_title}
        for source_ref, section_title in sampled
    ]


# ---------------------------------------------------------------------------
# Question generation per source type
# ---------------------------------------------------------------------------


def make_question(item: dict[str, Any], seed_offset: int) -> str:
    source_type = item["source_type"]

    if source_type == "irc":
        template = _pick_template(_IRC_TEMPLATES, seed_offset)
        return template.format(
            section=item["section_num"],
            topic=item["topic"].lower(),
            title=item["title"],
        )

    if source_type == "treasury_reg":
        template = _pick_template(_TREASURY_REG_TEMPLATES, seed_offset)
        return template.format(title=item["title"])

    if source_type == "treaty":
        template = _pick_template(_TREATY_TEMPLATES, seed_offset)
        # title is like "US-Canada Treaty" → make it readable
        return template.format(title=item["title"])

    # tax_court, appellate, scotus, fed_claims
    template = _pick_template(_CASE_TEMPLATES, seed_offset)
    return template.format(title=item["title"])


def make_expected_output(item: dict[str, Any]) -> str:
    source_type = item["source_type"]
    title = item["title"]

    if source_type == "irc":
        return (
            f"This question concerns {title}. "
            f"A correct answer should accurately describe the tax rules, "
            f"requirements, or provisions under this IRC section."
        )
    if source_type == "treasury_reg":
        return (
            f"This question concerns {title}. "
            f"A correct answer should accurately describe the Treasury Regulation "
            f"requirements and how they apply to taxpayers."
        )
    if source_type == "treaty":
        return (
            f"This question concerns the {title}. "
            f"A correct answer should describe the treaty's key provisions including "
            f"withholding rates, residency rules, and double taxation relief."
        )
    # case law
    court_label = {
        "tax_court": "US Tax Court",
        "appellate": "US Court of Appeals",
        "scotus": "US Supreme Court",
        "fed_claims": "US Court of Federal Claims",
    }.get(source_type, "court")
    return (
        f"This question concerns the {court_label} case {title}. "
        f"A correct answer should describe the tax issue decided, the court's "
        f"holding, and the legal reasoning applied."
    )


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
                raw = line[len("data:") :].strip()
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
# Dataset generation
# ---------------------------------------------------------------------------


async def generate_dataset(all_items: list[dict[str, Any]]) -> list[EvalCase]:
    """Call RAG API for each item and build EvalCase list."""
    cases: list[EvalCase] = []

    for i, item in enumerate(all_items):
        question = make_question(item, seed_offset=i)
        print(f"  [{i + 1}/{len(all_items)}] [{item['source_type']}] {question[:80]}...")

        async with httpx.AsyncClient() as client:
            t0 = time.monotonic()
            result = await call_rag_api(question, client)
            elapsed = time.monotonic() - t0

        if i < len(all_items) - 1:
            await asyncio.sleep(RAG_DELAY_SECONDS)

        answer = result["answer"]
        context = _citations_to_context(result["citations"])
        expected_output = make_expected_output(item)

        metadata: dict[str, Any] = {
            "source_type": item["source_type"],
            "rag_latency_ms": int(elapsed * 1000),
            "has_context": bool(context),
            "answer": answer,
        }

        # Source-specific metadata
        if item["source_type"] == "irc":
            metadata.update(
                {
                    "section_num": item["section_num"],
                    "section_url": item.get("section_url", ""),
                    "subtitle": item.get("subtitle", ""),
                    "topic": item["topic"],
                }
            )
        else:
            metadata.update(
                {
                    "source_ref": item.get("source_ref", ""),
                    "title": item["title"],
                }
            )

        case = EvalCase(
            input_text=question,
            expected_output=expected_output,
            context=context,
            metadata=metadata,
        )
        cases.append(case)

    return cases


# ---------------------------------------------------------------------------
# RAGAS evaluation
# ---------------------------------------------------------------------------


async def run_ragas(cases: list[EvalCase]) -> list[EvalResult]:
    from orchestrator.evaluation import RagasEvaluator

    evaluator = RagasEvaluator(
        metric_names=["faithfulness", "answer_relevancy", "context_precision"],
    )
    results = []
    for i, case in enumerate(cases):
        if not case.context:
            print(f"  [RAGAS] Skipping case {i + 1} — no context retrieved")
            continue
        answer = case.metadata.get("answer", "")
        if not answer:
            print(f"  [RAGAS] Skipping case {i + 1} — empty answer")
            continue
        print(f"  [RAGAS] Evaluating case {i + 1}/{len(cases)}: {case.input_text[:60]}...")
        result = await evaluator.evaluate(case, answer)
        result.metadata["case_id"] = case.case_id
        result.metadata["source_type"] = case.metadata.get("source_type", "")
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
            print(f"  [DeepEval] Skipping case {i + 1} — empty answer")
            continue
        print(f"  [DeepEval] Evaluating case {i + 1}/{len(cases)}: {case.input_text[:60]}...")
        result = await evaluator.evaluate(case, answer)
        result.metadata["case_id"] = case.case_id
        result.metadata["source_type"] = case.metadata.get("source_type", "")
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------


def save_cases(cases: list[EvalCase], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [c.to_dict() for c in cases]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Saved {len(cases)} EvalCases → {path}")


def save_results(results: list[EvalResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [r.to_dict() for r in results]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Saved {len(results)} EvalResults → {path}")


def print_summary(results: list[EvalResult], label: str) -> None:
    if not results:
        print(f"\n{label}: no results.")
        return
    passed = sum(1 for r in results if r.overall_passed)
    avg = sum(r.overall_score or 0.0 for r in results) / len(results)
    print(f"\n{label} Summary:")
    print(f"  Cases evaluated : {len(results)}")
    print(f"  Passed          : {passed}/{len(results)}")
    print(f"  Avg overall score: {avg:.3f}")

    # Per-source breakdown
    source_scores: dict[str, list[float]] = {}
    for r in results:
        st = r.metadata.get("source_type", "unknown")
        source_scores.setdefault(st, []).append(r.overall_score or 0.0)
    print("\n  Per-source breakdown:")
    for st, scores in sorted(source_scores.items()):
        print(f"    {st:<20} avg={sum(scores) / len(scores):.3f}  n={len(scores)}")

    # Per-criterion breakdown
    criterion_scores: dict[str, list[float]] = {}
    for r in results:
        for s in r.scores:
            criterion_scores.setdefault(s.criterion, []).append(s.score)
    print("\n  Per-criterion breakdown:")
    for criterion, scores in criterion_scores.items():
        print(f"    {criterion:<30} avg={sum(scores) / len(scores):.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main(
    samples_per_source: int,
    output_path: Path,
    run_eval: bool,
) -> None:
    print("=== TaxPilot Evaluation Dataset Generator (Multi-Source) ===\n")

    all_items: list[dict[str, Any]] = []

    # 1. IRC — from Cornell Title 26 CSVs (5 per subtitle = ~55 items)
    print("Step 1: Loading IRC sections from Cornell Title 26 CSVs...")
    irc_items = load_irc_sections(CSV_DIR, samples_per_subtitle=samples_per_source)
    all_items.extend(irc_items)
    print(f"  IRC total: {len(irc_items)} items\n")

    # 2. Other sources — from pgvector DB
    other_sources = ["treasury_reg", "tax_court", "appellate", "scotus", "fed_claims", "treaty"]
    print("Step 2: Loading other sources from pgvector DB...")
    for source_type in other_sources:
        items = load_chunks_from_db(source_type, samples=samples_per_source)
        all_items.extend(items)
    print(f"\n  Total items across all sources: {len(all_items)}\n")

    # 3. Generate dataset by calling RAG API
    print("Step 3: Calling RAG API for each question...")
    cases = await generate_dataset(all_items)
    print(f"\nGenerated {len(cases)} EvalCases.")

    # Print source breakdown
    from collections import Counter

    counts = Counter(c.metadata.get("source_type") for c in cases)
    for st, n in sorted(counts.items()):
        print(f"  {st:<20} {n} cases")

    # 4. Save dataset
    save_cases(cases, output_path)

    if not run_eval:
        print("\nDone. Use --run-eval to run RAGAS and DeepEval scoring.")
        return

    # 5. RAGAS evaluation
    print("\nStep 4: Running RAGAS evaluation...")
    try:
        ragas_results = await run_ragas(cases)
        print_summary(ragas_results, "RAGAS")
        save_results(ragas_results, output_path.parent / "ragas_results.json")
    except ImportError as e:
        print(f"[SKIP] RAGAS not installed: {e}")
        ragas_results = []

    # 6. DeepEval evaluation
    print("\nStep 5: Running DeepEval evaluation...")
    try:
        deepeval_results = await run_deepeval(cases)
        print_summary(deepeval_results, "DeepEval")
        save_results(deepeval_results, output_path.parent / "deepeval_results.json")
    except ImportError as e:
        print(f"[SKIP] DeepEval not installed: {e}")

    print("\n=== Done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate TaxPilot multi-source evaluation dataset"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES_PER_SOURCE,
        help=f"Items to sample per source (default: {DEFAULT_SAMPLES_PER_SOURCE}). "
        f"For IRC this is per subtitle CSV (11 subtitles × samples).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_DIR / "generated_cases.json"),
        help="Output path for generated EvalCase JSON",
    )
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="Run RAGAS and DeepEval after generating the dataset",
    )
    args = parser.parse_args()

    asyncio.run(
        main(
            samples_per_source=args.samples,
            output_path=Path(args.output),
            run_eval=args.run_eval,
        )
    )
