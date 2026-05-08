---
name: continuum-workflows
description: Use Continuum's nine workflow agents — Sequential, Parallel, Loop, Reflection, Router, Planner, Debate, Scatter, SupervisedSequential — to build multi-agent pipelines. Invoke when the user asks "chain agents", "run agents in parallel", "iterate until done", "self-improving agent", "route to specialist", "decompose a goal", or anything multi-agent.
---

# Continuum Workflow Agents Skill

Workflow agents are themselves `BaseAgent` subclasses, so they nest.
Each ships with a `create_*` factory.

Authoritative source: [`docs/agent.md`](../../../docs/agent.md), §6.

---

## Imports

```python
from orchestrator.agent import (
    create_sequential_agent, create_parallel_agent, create_loop_agent,
    create_reflection_agent, create_planner_agent, create_router_agent,
    MemoryScope, MergeStrategy, FailStrategy, TerminationType,
)
# Debate, Scatter, and Supervised factories live one level deeper —
# they are NOT re-exported from `orchestrator.agent`.
from orchestrator.agent.workflow import (
    create_debate_agent, create_scatter_agent, create_supervised_agent,
)
```

---

## SequentialAgent

```python
pipeline = create_sequential_agent(
    name="content-pipeline",
    agents=[researcher, writer, editor],
    pass_full_history=False,
    fail_strategy=FailStrategy.FAIL_FAST,
)
```

Use when each agent's output should feed the next.

---

## ParallelAgent

```python
fanout = create_parallel_agent(
    name="fanout",
    agents=[a, b, c],
    merge_strategy=MergeStrategy.LLM_SUMMARIZE,    # CONCATENATE / STRUCTURED / FIRST_SUCCESS
    fail_strategy=FailStrategy.CONTINUE_ON_ERROR,
    timeout=300,
)
```

Use when independent perspectives can run concurrently and you want
them merged.

---

## LoopAgent

```python
iterate = create_loop_agent(
    name="iterate-until-done",
    agent=worker,
    termination_type=TerminationType.LLM_DECISION,
    max_iterations=10,
    termination_prompt="Reply COMPLETE if done, CONTINUE otherwise.",
    # or termination_tool="finish" / termination_pattern="^DONE" /
    #     termination_condition=fn(content, history)
)
```

---

## ReflectionAgent

```python
self_improving = create_reflection_agent(
    name="self-improving",
    agent=writer,
    critique_prompt=None,                  # default critique
    max_reflections=2,
    reflection_model=None,                 # defaults to writer's model
)
```

`generate_critique_prompt(user_query, llm_client)` builds a
query-specific critique prompt programmatically.

---

## RouterAgent

```python
router = create_router_agent(
    name="triage",
    routes=[
        ("billing-agent",   "Billing & payment issues"),
        ("technical-agent", "Technical support"),
        ("sales-agent",     "Sales / pricing"),
    ],
    fallback="general-agent",
    strategy="hybrid",                    # "llm" | "rule_based" | "hybrid"
    model=None,
)
```

`Route` (not `("name", "desc")`) is `Route(agent_name=..., description=...)`.

---

## PlannerAgent

Two modes: single-worker, or agent-pool (LLM picks specialist per step).

```python
planner = create_planner_agent(
    name="planner",
    agents=[researcher, coder, reviewer],
    instructions="You are a planning agent.",
    max_steps=10,
    enable_replanning=False,
    replan_on_failure=True,
    fail_strategy=FailStrategy.FAIL_FAST,
    strict_agent_pool=False,
)
```

---

## DebateAgent

```python
debate = create_debate_agent(
    name="debate",
    topic_description="…",
    pro_stance="Argue in favor.",
    con_stance="Argue against.",
    judge_instructions=None,
    summarise_arguments=False,
    truncate_chars=2000,
)
```

Pro and con run in parallel; judge synthesizes.

---

## ScatterAgent

```python
scatter = create_scatter_agent(
    name="scatter",
    agents=[a, b, c],
    input_slices=None,                    # let the LLM split, or pass pre-cut slices
    merge_strategy=MergeStrategy.LLM_SUMMARIZE,
    fail_strategy=FailStrategy.CONTINUE_ON_ERROR,
    split_model=None,
    timeout=300,
)
```

LLM splits input into N focused sub-tasks (one per branch).

---

## SupervisedSequentialAgent

Sequential with LLM quality gating — retries any step that scores below
`quality_threshold`.

```python
supervised = create_supervised_agent(
    name="supervised",
    agents=[step1, step2, step3],
    quality_threshold=0.7,
    max_retries=2,
    supervisor_model=None,
)
```

---

## Nesting

Workflow agents are `BaseAgent` subclasses, so nest freely:

```python
inner_pipeline = create_sequential_agent(name="inner", agents=[a, b])
outer = create_parallel_agent(name="outer", agents=[inner_pipeline, c, d])
```

---

## Don't

- Don't use `MergeStrategy.LLM_SUMMARIZE` without setting a sensible
  `summary_model` for cost reasons — defaults to the parent agent's model.
- Don't expect `RouterAgent` to dispatch to agents that aren't
  registered with the runner via `register_agent()`.
- Don't pass `Route(target=..., description=...)` — it's `agent_name=...`.
- Don't loop forever — `LoopAgent` always needs a termination strategy.
