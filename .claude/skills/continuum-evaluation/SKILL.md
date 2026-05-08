---
name: continuum-evaluation
description: Evaluate agent quality with the `EvaluatorAgent`, build golden datasets from Langfuse traces, and run DeepEval/RAGAS metrics over conversations. Invoke when the user asks "test agent quality", "evaluate output", "RAG metrics", "DeepEval", "RAGAS", or "regression-test my agent".
---

# Continuum Evaluation Skill

Continuum ships an evaluation framework under `orchestrator.evaluation`.
It's an **optional extra** — install with:

```bash
pip install "shyftlabs-continuum[eval]"
# adds: deepeval >= 1.0.0, ragas >= 0.2.0
```

Authoritative source: `src/orchestrator/evaluation/` in this
repository. There is no dedicated user-facing doc; this skill is the
primary reference.

---

## Imports

```python
from orchestrator.evaluation import (
    EvaluatorAgent,                   # specialised agent that scores other agents
    DeepEvalEvaluator,                # DeepEval criterion-based evaluation
    RagasEvaluator,                   # RAGAS metrics for RAG pipelines
    build_golden_dataset,             # build dataset from Langfuse traces
    generate_eval_dataset,            # synthesize evaluation cases from a domain spec
)
```

---

## EvaluatorAgent

A specialised `BaseAgent` that takes another agent's output (plus
optional reference) and produces a structured score.

```python
from orchestrator.agent import BaseAgent, AgentRunner
from orchestrator.evaluation import EvaluatorAgent

target_agent = BaseAgent(name="target", instructions="Answer concisely.")
evaluator = EvaluatorAgent(
    name="quality-judge",
    criteria=[
        "Factual accuracy (0-1)",
        "Conciseness (0-1)",
        "Helpfulness (0-1)",
    ],
    model="gpt-4o-mini",
)

# Evaluate one trace
target_resp = await AgentRunner().run(target_agent, "What is the capital of France?")
score = await evaluator.evaluate(
    input="What is the capital of France?",
    output=target_resp.content,
    reference="Paris is the capital of France.",
)
print(score)         # {"factual_accuracy": 1.0, "conciseness": 0.9, "helpfulness": 0.8}
```

---

## DeepEval

Use for criterion-based per-case scoring; great for unit-test-style
agent regression suites.

```python
from orchestrator.evaluation import DeepEvalEvaluator
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

evaluator = DeepEvalEvaluator(
    metrics=[
        AnswerRelevancyMetric(threshold=0.7),
        FaithfulnessMetric(threshold=0.8),
    ],
)

results = await evaluator.evaluate_batch([
    {"input": "...", "output": "...", "expected_output": "...", "context": ["..."]},
    # …
])
for r in results:
    print(r.metric_name, r.score, r.success, r.reason)
```

---

## RAGAS

For RAG-specific metrics (faithfulness, context precision/recall,
answer correctness, etc.).

```python
from orchestrator.evaluation import RagasEvaluator
from ragas.metrics import faithfulness, answer_relevancy, context_precision

evaluator = RagasEvaluator(
    metrics=[faithfulness, answer_relevancy, context_precision],
)

dataset = [
    {
        "question": "What is the capital of France?",
        "answer": agent_output,
        "contexts": retrieved_docs,           # the chunks fed into the prompt
        "ground_truth": "Paris",
    },
]
report = await evaluator.evaluate(dataset)
```

---

## Build a golden dataset from Langfuse traces

```python
from orchestrator.evaluation import build_golden_dataset

dataset = await build_golden_dataset(
    project="my-project",
    tags=["production", "sampled"],
    limit=200,
    after="2026-01-01",
    user_label_field="thumbs_up",       # only keep traces the user labeled positively
)
```

The returned object is a list of `{"input", "output", "metadata"}`
dicts you can feed straight into DeepEval / RAGAS.

`generate_eval_dataset(domain_spec, n=50, model="gpt-4o-mini")` is the
synthetic counterpart — useful when you don't yet have production
traffic.

---

## Putting it together

```python
async def regression_check(agent, dataset, threshold=0.7):
    runner = AgentRunner()
    evaluator = DeepEvalEvaluator(metrics=[AnswerRelevancyMetric(threshold=threshold)])

    cases = []
    for sample in dataset:
        resp = await runner.run(agent, sample["input"])
        cases.append({
            "input": sample["input"],
            "output": resp.content,
            "expected_output": sample.get("expected_output"),
        })

    results = await evaluator.evaluate_batch(cases)
    failures = [r for r in results if not r.success]
    return failures
```

Wire this into CI to catch regressions before they ship.

---

## Don't

- Don't run RAGAS without `contexts` populated — most RAGAS metrics
  need the retrieved chunks.
- Don't reuse `EvaluatorAgent` across hostile production traffic — its
  outputs are LLM-generated and can be cheap to game; treat scores as
  signals, not gates.
- Don't compare scores across different judge models / prompts —
  always pin the evaluator's `model` and `criteria` for stable
  longitudinal comparisons.
- Don't forget `pip install "shyftlabs-continuum[eval]"` (or
  `pip install -e ".[eval]"` from a checkout) before importing
  `orchestrator.evaluation` — the eval module's third-party deps are
  an optional extra.
