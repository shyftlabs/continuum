# Temporal — Custom Workflows & Activities

The built-in workflows cover most patterns, but you can write your own
`@workflow.defn` and `@activity.defn` and have the worker pick them up
automatically.

---

## 1 · Custom workflow

Workflow code runs inside Temporal's deterministic sandbox. **No I/O
allowed** — every side effect must go through an activity.

```python
from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from orchestrator.temporal.types import (
        AgentActivityParams, AgentActivityResult, WorkflowResult,
    )

@workflow.defn(sandboxed=False)
class TranslationFanoutWorkflow:
    """
    Run the same input through multiple translators in parallel,
    then have a reviewer agent choose the best.
    """

    @workflow.run
    async def run(self, input_text: str) -> WorkflowResult:
        translators = ["translator_french", "translator_german", "translator_spanish"]
        retry = RetryPolicy(maximum_attempts=3, initial_interval=timedelta(seconds=1),
                            backoff_coefficient=2.0)

        # Fan out
        handles = [
            workflow.start_activity(
                "run_agent_activity",
                AgentActivityParams(agent_name=name, input=input_text),
                start_to_close_timeout=timedelta(seconds=300),
                retry_policy=retry,
            )
            for name in translators
        ]
        results: list[AgentActivityResult] = list(await asyncio.gather(*handles))

        # Pick the best
        chooser = await workflow.execute_activity(
            "run_agent_activity",
            AgentActivityParams(
                agent_name="best_translation_picker",
                input="\n\n".join(f"{t.agents_used[0]}:\n{t.content}" for t in results),
            ),
            start_to_close_timeout=timedelta(seconds=120),
            retry_policy=retry,
        )

        return WorkflowResult(
            status="completed",
            content=chooser.content,
            step_results=results + [chooser],
        )
```

### Register with the worker

```python
from orchestrator.temporal import get_worker_manager, get_temporal_client

worker = get_worker_manager(get_temporal_client(), get_agent_registry())
worker.register_workflow(TranslationFanoutWorkflow)
await worker.start()
```

`register_workflow()` adds your class to the worker's workflow list
*before* it starts. The worker then accepts both built-in workflows
(`AgentWorkflow`, `SequentialAgentWorkflow`, …) and yours.

### Submit it

```python
client = get_temporal_client()
await client.connect()
handle = await client.start_workflow(
    TranslationFanoutWorkflow.run,
    "Hello world",
    id="translation-001",
    task_queue="orchestrator-agents",
)
result = await handle.result()
```

---

## 2 · Custom activity

Activities are where I/O lives. Anything async-friendly works:
filesystem, HTTP, DB, queues, your own internal services.

```python
from temporalio import activity
import httpx

@activity.defn(name="fetch_url_activity")
async def fetch_url_activity(url: str) -> str:
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.text
```

### Register with the worker

```python
worker.register_activity(fetch_url_activity)
```

### Use from a workflow

```python
@workflow.defn(sandboxed=False)
class FetchAndSummarizeWorkflow:
    @workflow.run
    async def run(self, url: str) -> WorkflowResult:
        text = await workflow.execute_activity(
            fetch_url_activity,
            url,
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )
        summary = await workflow.execute_activity(
            "run_agent_activity",
            AgentActivityParams(agent_name="summarizer", input=text),
            start_to_close_timeout=timedelta(seconds=120),
        )
        return WorkflowResult(status="completed", content=summary.content)
```

---

## 3 · Determinism rules (read this!)

Workflow code is replayed during recovery. Anything that returns a
different result on replay breaks Temporal's contract.

**Don't:**
- `random.random()`, `uuid.uuid4()` — use `workflow.uuid4()` and `workflow.random()`
- `datetime.now()`, `time.time()` — use `workflow.now()`
- `requests.get()`, file I/O, DB calls — call activities instead
- Spawn threads or asyncio tasks outside `workflow.start_activity` /
  `workflow.start_child_workflow`
- Run import-time side effects unless wrapped in
  `with workflow.unsafe.imports_passed_through():`

**Do:**
- `await workflow.sleep(...)`, `await workflow.wait_condition(...)`
- `workflow.signal`, `workflow.query` decorators
- `workflow.execute_activity(...)`, `workflow.start_activity(...)`
- Keep the workflow body small — push complexity into activities

---

## 4 · Signals & queries

```python
@workflow.defn(sandboxed=False)
class MyWorkflow:
    def __init__(self):
        self._cancel = False
        self._injected_input: str | None = None

    @workflow.signal
    async def cancel_workflow(self) -> None:
        self._cancel = True

    @workflow.signal
    async def inject_input(self, data: str) -> None:
        self._injected_input = data

    @workflow.query
    def get_status(self) -> dict[str, Any]:
        return {"cancelled": self._cancel, "injected": self._injected_input is not None}

    @workflow.run
    async def run(self) -> str:
        await workflow.wait_condition(lambda: self._injected_input or self._cancel)
        if self._cancel:
            return "cancelled"
        return f"got: {self._injected_input}"
```

Signal from outside:

```python
await client.signal_workflow("workflow-id", "inject_input", "hello")
```

Query from outside:

```python
status = await client.query_workflow("workflow-id", "get_status")
```

---

## 5 · Patterns

### Long-running with periodic heartbeat

```python
@activity.defn
async def long_running_activity(params: dict) -> str:
    while not done():
        activity.heartbeat({"progress": progress()})
        await asyncio.sleep(5)
    return result
```

Set `heartbeat_timeout=timedelta(seconds=30)` on the call site.

### Fan-out with bounded concurrency

```python
async def run(self, items: list[str]) -> list[str]:
    semaphore = asyncio.Semaphore(5)
    async def one(item):
        async with semaphore:
            return await workflow.execute_activity(
                "run_agent_activity", AgentActivityParams(agent_name="worker", input=item),
                start_to_close_timeout=timedelta(seconds=60),
            )
    return await asyncio.gather(*(one(i) for i in items))
```

### Child workflow

```python
result = await workflow.execute_child_workflow(
    OtherWorkflow.run, child_input, id="child-123",
)
```

---

## 6 · Common errors

| Error | Cause |
|---|---|
| `Activity is not registered` | You forgot `worker.register_activity(fn)` |
| `Workflow is not registered` | You forgot `worker.register_workflow(cls)` |
| `non-deterministic` errors during replay | I/O, randomness, or `datetime.now()` in workflow body |
| `task_queue` mismatch | Caller and worker use different `TEMPORAL_TASK_QUEUE` |
| `WorkflowAlreadyStartedError` | Re-using the same workflow `id` |
