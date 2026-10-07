# System One Module

A **System One classifier** answers typed questions about a state with a
probability per outcome: fast, cheap judgement, in Kahneman's sense, next to
the slow reasoning of an LLM. Continuum uses one, where you opt in, for four
decisions an LLM call makes today:

| Seam | Question asked | Turn it on with |
|---|---|---|
| `RouterAgent` | which route fits this request (or none)? | `RouterConfig(routing_strategy="system_one_classifier")` |
| `LoopAgent` | is the task complete? | `TerminationConfig(type=TerminationType.SYSTEM_ONE_CLASSIFIER)` |
| `ReflectionAgent`, `SupervisedSequentialAgent` | does this draft fully answer the request? | `verdict_mode="system_one_classifier"` |
| tool approval | would this tool call be risky? | `approval_handler=system_one_approval_handler(...)` |

**Nothing is on by default.** Configuring a backend enables no seam; each seam
opts in on its own, and with its switch off it behaves exactly as before.
`SYSTEM_ONE_DISABLED=true` sends every opted-in seam back to that behaviour
without a code change.

What this module gives you:
- `classify()` — one call for every backend: kill switch, egress policy check,
  validation and normalisation
- three question types and their answers, each answer carrying the backend's
  own `confidence`
- five built-in backends, and an entry-point group for third-party ones
- `SystemOneContract` — the test suite a backend must pass

---

## 1 · Getting started

Three steps: pick a backend, put two lines in `.env`, and switch System One on
in code where you want it.

### 1.1 · Pick a backend

| You want | `SYSTEM_ONE_BACKEND` | Install |
|---|---|---|
| Jev, hosted (best results in our tests) | `openrouter:typesafe/jev-1.13`, or `jev:jev-latest` direct from TypeSafe | nothing extra |
| everything on your own machine | `laya:convaiinnovations/laya` | `pip install "shyftlabs-continuum[laya]"` |
| on your own machine, Apple Silicon | `laya-mlx:aac6fef/laya-mlx` | `pip install "shyftlabs-continuum[laya-mlx]"` |
| a small local model | `local:cross-encoder/nli-deberta-v3-small` | `pip install "shyftlabs-continuum[embeddings]"` |

### 1.2 · Set it in `.env`

One required line, plus the key for that backend — and only that one.
Everything else is optional.

```bash
# Required: the backend. Setting it switches nothing on by itself.
SYSTEM_ONE_BACKEND=openrouter:typesafe/jev-1.13

# The key for that backend:
OPENROUTER_API_KEY=sk-or-...          # openrouter:...
# TYPESAFE_API_KEY=...                # jev:...
# local:, laya: and laya-mlx: need no key

# Optional:
# SYSTEM_ONE_TIMEOUT_SECONDS=10       # per backend call
# SYSTEM_ONE_DISABLED=true            # kill switch: seams go back to the LLM / a person
# OPENROUTER_BASE_URL=...             # only to override the endpoint
# TYPESAFE_BASE_URL=...               # only to override the endpoint
```

### 1.3 · Switch it on in code

Each seam opts in on its own; turn on only the ones you want. The backend comes
from `SYSTEM_ONE_BACKEND` unless a seam names its own.

```python
from continuum.agent import (
    AgentConfig, LoopAgent, ReflectionConfig, Route, RouterAgent,
    RouterConfig, TerminationConfig, TerminationType,
)
from continuum.agent.system_one_approval import system_one_approval_handler
from continuum.agent.workflow.supervised import SupervisedConfig

# Router: the classifier picks the route. Below the floor, the request goes to the fallback,
# so make the fallback an agent that asks the user what they meant.
router = RouterAgent(
    name="triage",
    routes=[
        Route(agent_name="billing-agent", description="payments, invoices and refunds"),
        Route(agent_name="technical-agent", description="software errors, bugs and outages"),
    ],
    fallback_agent_name="clarify-agent",
    router_config=RouterConfig(
        routing_strategy="system_one_classifier",
        system_one_min_confidence=0.5,     # optional; measured for Jev
    ),
)

# Loop: stop when P(complete) >= 0.5
loop = LoopAgent(
    name="refine",
    agent=writer,
    termination=TerminationConfig(type=TerminationType.SYSTEM_ONE_CLASSIFIER, max_iterations=5),
)

# Reflection / supervised: a confident pass skips the critic or supervisor call
ReflectionConfig(verdict_mode="system_one_classifier")
SupervisedConfig(verdict_mode="system_one_classifier")

# Tool approval: clearly low-risk calls to listed tools are approved; everything else asks a person
AgentConfig(
    tool_approval={"get_*", "send_*"},
    approval_handler=system_one_approval_handler(
        auto_approve_tools={"get_*"},
        escalate_to=your_reviewer,
    ),
)
```

### 1.4 · Is it working?

| You see | It means |
|---|---|
| One log line per decision with `decided_by=system_one`, e.g. `Router 'triage' decided_by=system_one backend=openrouter route=billing-agent p=0.990 confidence=0.990`, `Loop termination decided_by=system_one … p_complete=0.920 threshold=0.5 -> stop`, `Tool 'get_balance' auto-approved by system_one:… (P(risky)=0.040 < 0.1)` | Working |
| `SystemOneNotConfiguredError` raised when the agent is built | No backend: set `SYSTEM_ONE_BACKEND` |
| The agent builds, but every decision logs a `WARNING` such as `System One routing failed … (SystemOneNotConfiguredError)` | The backend cannot be built — usually its key or its pip extra is missing. Each seam falls back (§5) |
| A `WARNING` naming `SystemOneTimeoutError` or `SystemOneBackendError` | The backend is slow or down; each seam falls back the same way |
| A log line naming `SYSTEM_ONE_DISABLED`, e.g. `Loop termination decided_by=legacy (SYSTEM_ONE_DISABLED)` | The kill switch is on. Tool approval logs nothing extra: every gated call simply goes to the person |

> **Three things that catch newcomers.** (1) `SYSTEM_ONE_BACKEND` alone changes
> nothing — a seam must also be switched on in code. (2) A local backend's first
> call is slow: it loads the model once. (3) Thresholds do not carry over between
> backends: the router floor of 0.5 suits Jev but sends most Laya requests to the
> fallback, and local NLI reports no confidence at all, so with a floor it never
> routes.

---

## 2 · Backends

A backend is named by a spec, `<prefix>:<model>`.

| Prefix | Backend | Example spec | Runs | Question types | Needs |
|---|---|---|---|---|---|
| `jev:` | TypeSafe's Jev, direct | `jev:jev-latest` | remote | binary, choice, score | `TYPESAFE_API_KEY` |
| `openrouter:` | Jev through OpenRouter's Decisions API | `openrouter:typesafe/jev-1.13` | remote | binary, choice, score | `OPENROUTER_API_KEY` |
| `local:` | an NLI cross-encoder | `local:cross-encoder/nli-deberta-v3-small` | this process | binary, choice | `[embeddings]` extra |
| `laya:` | Convai's open-weight Laya | `laya:convaiinnovations/laya` | this process | binary, choice, score | `[laya]` extra |
| `laya-mlx:` | Laya on Apple Silicon MLX | `laya-mlx:aac6fef/laya-mlx` | this process | binary, choice, score | `[laya-mlx]` extra |

- **Each backend reads only its own key.** A TypeSafe key is never sent to
  OpenRouter, or the reverse. A backend without its key (or a local backend
  without its package) raises `SystemOneNotConfiguredError`, naming what to set,
  when it is first built — at the first decision, not when the agent is built —
  and each seam treats that as a failure (§5).
- **Remote backends retry once** on 408, 429, 500, 502–504, 524 and 529, honouring
  `Retry-After` but never past the call's timeout. Cost and token usage are in
  `provenance.usage`.
- **Local backends load lazily**, once, on the first call. Laya silently cuts
  state longer than its context; the adapter reports it as
  `usage["truncated"] = True` and logs it, never the state itself.
- **A question type a backend lacks is filled in** by the layer: a score is
  asked as a choice over its levels, a choice as one binary question per label.
  The answer is flagged `filled_in=True`, because a derived distribution may be
  less well calibrated than a native one.

There is deliberately no `llm:` prefix. A general-purpose LLM is not a System
One backend: with a seam's switch off, that seam already uses the LLM.

### Choosing the backend

Each seam takes its own spec (`system_one_backend=` / `backend=`); `None`, the
default, uses the container's backend, which is built from
`SYSTEM_ONE_BACKEND` on first use. `container.set_system_one_classifier(obj)`
injects one directly.

**A seam that opts in with no backend fails at construction**, not at its first
decision: `SystemOneNotConfiguredError`. Otherwise you would believe the
classifier was deciding when nothing was.

### Third-party backends

A backend Continuum does not ship can be added in its own package, with no
change to the SDK: once installed, its spec (`mybackend:<model>`) works
anywhere a built-in one does. See §9.

---

## 3 · `classify()`

```python
from continuum.system_one import BinaryQuestion, ChoiceQuestion, ScoreQuestion, classify

resp = await classify(
    "I was charged twice for my subscription, please refund me.",
    {
        "urgent": BinaryQuestion(
            instructions="Does this message express urgency?",
            true_criteria="The sender needs help now.",
            false_criteria="There is no time pressure.",
        ),
        "team": ChoiceQuestion(
            instructions="Which team should handle this?",
            labels={"billing": "Payments, invoices, refunds", "technical": "Bugs and outages"},
        ),
        "frustration": ScoreQuestion(
            instructions="How frustrated is the sender?",
            levels=["Calm.", "Concerned.", "Very frustrated."],
        ),
    },
    spec="openrouter:typesafe/jev-1.13",   # optional; default as in §2
)

resp.binary("urgent").probability        # P(true), 0..1
resp.choice("team").label                # the most likely label
resp.choice("team").probabilities        # {"billing": 0.97, "technical": 0.03}
resp.score("frustration").expected       # expected level index, 0..2
resp.provenance.backend, resp.provenance.model, resp.provenance.latency_ms
```

`state` is a string or a JSON-serialisable object; a text-only backend receives
JSON state as JSON text. `classify(state, questions, *, classifier=None,
spec=None)` uses `classifier` as given, else resolves `spec`, else the default.

Between the question and the answer, the layer:

1. **refuses when `SYSTEM_ONE_DISABLED` is set** (`SystemOneDisabledError`), so
   a direct call cannot bypass the switch;
2. **checks egress policy** before any state leaves the process (§6);
3. fills in question types the backend lacks (§2);
4. **validates every distribution** — the right outcome keys, finite,
   non-negative, not all zero — and normalises it.

Whenever no trusted answer exists it raises a `SystemOneError` subclass; it
never guesses. Each seam decides what that means for it (§5).

| Exception | When |
|---|---|
| `SystemOneNotConfiguredError` | no backend, a bad spec, or a backend missing its key or package |
| `SystemOneDisabledError` | `SYSTEM_ONE_DISABLED` is set |
| `SystemOneAccessDeniedError` | the run's policy denies this backend's resource (§6) |
| `SystemOneTimeoutError` | no answer within `SYSTEM_ONE_TIMEOUT_SECONDS` |
| `SystemOneBackendError` | the backend failed (HTTP error, model error) |
| `SystemOneResponseError` | the answer is missing or malformed |
| `SystemOneCapabilityError` | a question type the backend cannot answer, even filled in |

---

## 4 · Answers and confidence

| Question | Answer | Fields |
|---|---|---|
| `BinaryQuestion(instructions, true_criteria=None, false_criteria=None)` | `BinaryAnswer` | `probability`, `confidence`, `filled_in` |
| `ChoiceQuestion(instructions, labels)` | `ChoiceAnswer` | `label`, `probabilities`, `confidence`, `filled_in` |
| `ScoreQuestion(instructions, levels)` | `ScoreAnswer` | `expected`, `probabilities` (by level index), `confidence`, `filled_in` |

**`confidence` is the backend's own, passed through unchanged.** Continuum
computes no confidence of its own. It is `None` when the backend reports none,
and for a filled-in answer, whose per-question figures do not describe the
combined answer.

| Backend | Binary | Choice | Score |
|---|---|---|---|
| Jev (`jev:`, `openrouter:`) | none — Jev reports none for a yes/no | reported | reported |
| Laya (`laya:`, `laya-mlx:`) | reported | reported | reported |
| Local NLI (`local:`) | none | none | none |

TypeSafe's guidance for Jev: *the answer tells you what; confidence tells you
whether to act.* The router's floor (§5.1) is the one place the SDK acts on it.

**A threshold does not carry over between backends.** Each reports
probabilities and confidence on its own scale. For the same request, "show me
dog toys", Jev's route confidence was 1.00 and Laya's 0.03. Set a threshold for
the backend you measured it on.

---

## 5 · The seams

Every seam takes a backend spec of its own, logs each decision with
`decided_by=system_one`, and falls back to its previous behaviour under
`SYSTEM_ONE_DISABLED`:

| Seam | Decides | Default threshold | When the backend fails | `SYSTEM_ONE_DISABLED` |
|---|---|---|---|---|
| router | the top route | none (`system_one_min_confidence=None`) | no route → `fallback_agent_name` | LLM routing |
| loop | stop at P(complete) ≥ threshold | `system_one_threshold=0.5` | the LLM check | the LLM check |
| quality gate | pass at P(pass) ≥ threshold; may only **pass** | `system_one_pass_threshold=0.9` | the LLM critic / supervisor | the LLM critic / supervisor |
| tool approval | approve at P(risky) < threshold; may only **approve** | `auto_approve_below=0.1` | a person | a person |

### 5.1 · `RouterAgent`

```python
RouterConfig(
    routing_strategy="system_one_classifier",
    system_one_backend=None,          # None = SYSTEM_ONE_BACKEND
    system_one_min_confidence=None,   # e.g. 0.5; None = act on the top route
)
```

The backend answers a choice over the route names plus `"none"`. Nothing is
parsed: the answer is one of the offered labels. Each label is the route's
description as a statement ("This request is about payments, invoices and
refunds."), so an NLI backend, which judges premise and hypothesis only, can
test it.

- `"none"` selects no route, and the request goes to `fallback_agent_name`.
- **With `system_one_min_confidence` set**, a top route whose confidence is
  below it is not acted on: no route, as for `"none"`. A backend that reports
  no confidence (local NLI) cannot meet a floor, so it always gets no route,
  with a `WARNING`.
- **A failed call** (down, timed out, denied by policy) is also no route. The
  router does not fall back to the LLM: that would put a second, unrequested
  decision-maker behind the one you chose.

Give the router a fallback that asks rather than guesses: below the floor the
request was, by definition, not clear.

### 5.2 · `LoopAgent`

```python
from continuum.agent import LoopAgent, TerminationConfig, TerminationType

LoopAgent(
    name="iterate-until-done",
    agent=worker,
    termination=TerminationConfig(
        type=TerminationType.SYSTEM_ONE_CLASSIFIER,
        max_iterations=5,
        system_one_backend=None,      # None = SYSTEM_ONE_BACKEND
        system_one_threshold=0.5,     # stop when P(complete) >= this
    ),
)
```

The state sent is the task and the latest output, so "is this complete?" is
asked about the request, not the output alone. A failed call uses the LLM
check for that iteration.

### 5.3 · `ReflectionAgent` and `SupervisedSequentialAgent`

```python
from continuum.agent import ReflectionConfig
from continuum.agent.workflow.supervised import SupervisedConfig

ReflectionConfig(verdict_mode="system_one_classifier", system_one_pass_threshold=0.9)
SupervisedConfig(verdict_mode="system_one_classifier", system_one_pass_threshold=0.9)
```

A fast path in front of the LLM judge. The backend is asked whether the draft
fully and correctly answers the request. At P(pass) ≥ the threshold the draft
passes with no critic or supervisor call. **Anything else** — a lower score, a
failed call, the kill switch — goes to the LLM judge exactly as without the
classifier. So a fail and its feedback always come from the LLM; the one new
risk is a confident false pass, which the conservative 0.9 default is set
against.

### 5.4 · Tool approval: `system_one_approval_handler`

```python
from continuum.agent.system_one_approval import system_one_approval_handler

AgentConfig(
    tool_approval={"get_*", "transfer_*", "send_*"},
    tool_data_labels={"fetch_url": {"untrusted"}},       # what rule 1 reads
    approval_handler=system_one_approval_handler(
        auto_approve_tools={"get_*"},                    # rule 2: required
        no_auto_approve_with_labels={"*"},               # rule 1: the default
        escalate_to=human_reviewer,                      # everything else
        auto_approve_below=0.1,
        backend=None,                                    # None = SYSTEM_ONE_BACKEND
    ),
)
```

It plugs into the existing gate ([tools.md §6.6](tools.md#66--tool-approval-human-in-the-loop))
and has three outcomes: auto-approve, escalate to `escalate_to`, or — with no
reviewer — refuse. A call is auto-approved only if all hold, in this order:

1. **rule 2** — the tool matches `auto_approve_tools`;
2. **rule 1** — the run carries no data label matching
   `no_auto_approve_with_labels`;
3. the backend scores P(risky) strictly below `auto_approve_below`.

Rules 2 and 1 are deterministic and run before anything is sent, so a call they
stop never leaves the process, and text in an email the agent read cannot argue
with them:

- **Rule 2 caps what a fooled classifier can approve**: at most a tool you named.
  It is required.
- **Rule 1 stops auto-approval once the run has read labelled content.** It
  reads the labels you declare (`tool_data_labels`, memory `scope_data_labels`,
  run-level `data_labels`); the SDK ships no detector, and an agent that
  declares none is warned once that the rule has nothing to read.

The classifier sees the tool name, its arguments, the agent and the run's data
labels — not the conversation. It cannot tell a call the user asked for from
the same call an injected instruction produced; the rules are what cover that.

**It fails closed.** A backend that cannot answer, a policy denial and
`SYSTEM_ONE_DISABLED` all go to the person, or are refused with no person.
Nothing turns "no answer" into "approved". The default threshold is 0.1, not
0.5: on a backend whose scale you have not measured, "approve only when clearly
low" degrades into more prompts for a reviewer, never into a risky call
approved silently.

---

## 6 · Security

**Every call is a policy check.** Before any state is sent, the run's policy
store is asked about the resource

```
system_one:<egress>:<backend>:<model>      e.g. system_one:remote:openrouter:typesafe/jev-1.13
```

the same way an LLM call is checked against `llm:<model>`. `egress` is `remote`
or `local`, so a `phi`-labelled run can be denied every remote backend and
still use a local one:

```python
AccessPolicy(
    name="phi-no-remote-system-one",
    subjects=["phi"],
    resources=["system_one:remote:*"],
    effect="deny",
)
```

A denied call raises `SystemOneAccessDeniedError`, which each seam handles as
a failure (§5). With a `PolicyStore.default_deny(...)` store, allow the exact
resource of the backend you use, or every call is denied.
`egress_resource(classifier)` returns it.

**The kill switch** (`SYSTEM_ONE_DISABLED`) is read live — flipping it needs no
restart — and refuses in `classify()` too.

---

## 7 · Settings

| Variable | Default | Description |
|---|---|---|
| `SYSTEM_ONE_BACKEND` | unset | The default backend spec, e.g. `openrouter:typesafe/jev-1.13`. Enables nothing by itself |
| `SYSTEM_ONE_DISABLED` | `false` | Kill switch: every seam returns to its previous behaviour |
| `SYSTEM_ONE_TIMEOUT_SECONDS` | `10.0` | Per backend call |
| `TYPESAFE_API_KEY` | unset | The `jev:` backend |
| `TYPESAFE_BASE_URL` | `https://api.typesafe.ai` | |
| `OPENROUTER_API_KEY` | unset | The `openrouter:` backend |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api` | |

Local backends need their extra: `pip install "shyftlabs-continuum[laya]"`,
`"[laya-mlx]"` (Apple Silicon only) or `"[embeddings]"` for `local:`.

---

## 8 · Examples

Two playground projects show System One working end to end, each with a
browser UI where you can switch it on and off and watch every decision. Both
read `SYSTEM_ONE_BACKEND` and its key from the project-root `.env`; without a
usable backend their System One controls are greyed out with the reason.

| Project | Shows | Turn it on |
|---|---|---|
| [`playground/gateway-multi-agent-shop`](../playground/gateway-multi-agent-shop) | The **router, loop, reflection and supervised** seams on a pet shop: each System One mode is the plain workflow with System One switched on, and every reply ends with what System One decided for it — route and confidence, P(complete) per round, P(pass) per draft next to the critic's or supervisor's verdict. | Pick `router-system-one`, `loop-system-one`, `reflection-system-one` or `supervised-system-one` in the mode dropdown. |
| [`playground/data-label-clinic`](../playground/data-label-clinic) | **Tool approval** on a clinic agent: a harmless interaction check auto-approved, an email to an outside address sent to a person, and a patient-data run sent to a person without the classifier being asked. Each decision appears in the gate panel. | Tick **System One approval** next to Send. The three ways to watch it work are in [docs/system-one-approval.md](../playground/data-label-clinic/docs/system-one-approval.md). |

---

## 9 · Adding a backend

A backend Continuum does not ship — **local** (a model running in your
process, like Laya) or **remote** (a service you call, like Jev) — can live in
its own installable Python package, with no change to the SDK. Four steps:
write a class, register it, name it in a spec, and run the contract suite
against it. The backend only turns questions into **raw** probabilities; the
SDK does the rest — the kill switch, the policy check, validation and
normalisation, and filling in question types the backend does not answer
natively. The steps are the same for both; a remote backend also has to
handle the network (§9.4).

### 9.1 · The class

| Member | What it is |
|---|---|
| `name` | the spec prefix, e.g. `"coinflip"` for `coinflip:<model>` |
| `model` | the model this instance answers with |
| `capabilities` | `SystemOneCapabilities(question_types=..., egress=...)`: the kinds it answers natively (`"binary"`, `"choice"`, `"score"`), and `"local"` or `"remote"` |
| `async classify(state, questions)` | returns `SystemOneRawResult` |

`SystemOneRawResult(distributions=..., raw_confidence=..., model=..., usage=...)`:

- `distributions` — one per question ID: `{"true": p, "false": 1 - p}` for a
  binary question, `{label: p}` for a choice, `{level_index: p}` for a score;
- `raw_confidence` — the backend's own confidence per question ID, if it has one.
  It becomes the answer's `confidence` (§4); leave it out otherwise;
- `usage` — cost or token counts, shown in `provenance.usage`.

Two rules the contract suite checks:

- **refuse a question type you did not declare** with `SystemOneCapabilityError`.
  The SDK fills in undeclared types before they reach you, so one arriving is a
  bug;
- **every failure is a `SystemOneError` subclass.** The seams fall back only on
  those (§5); any other exception escapes them.

A minimal local backend:

```python
from continuum.system_one import (
    BinaryQuestion,
    ChoiceQuestion,
    SystemOneBackendError,
    SystemOneCapabilities,
    SystemOneCapabilityError,
    SystemOneRawResult,
)


class CoinFlipClassifier:
    name = "coinflip"  # the spec prefix: coinflip:<model>
    capabilities = SystemOneCapabilities(
        question_types=frozenset({"binary", "choice"}),  # score is filled in by the SDK
        egress="local",  # "remote" if it calls a service
    )

    def __init__(self, model: str = "v1") -> None:
        self.model = model

    async def classify(self, state, questions):
        for qid, q in questions.items():  # refuse kinds you did not declare
            if q.kind not in self.capabilities.question_types:
                raise SystemOneCapabilityError(
                    f"coinflip cannot answer a {q.kind} question ('{qid}')"
                )
        try:
            dists = {}
            for qid, q in questions.items():
                if isinstance(q, BinaryQuestion):
                    dists[qid] = {"true": 0.5, "false": 0.5}
                elif isinstance(q, ChoiceQuestion):
                    dists[qid] = {label: 1 / len(q.labels) for label in q.labels}
            return SystemOneRawResult(distributions=dists, model=self.model)
        except Exception as e:  # every failure -> a SystemOneError
            raise SystemOneBackendError(
                f"coinflip failed ({type(e).__name__})", backend=self.name
            ) from e


def make_classifier(model: str) -> CoinFlipClassifier:  # what the entry point names
    return CoinFlipClassifier(model=model)
```

### 9.2 · Register it, and use it

In the backend package's `pyproject.toml`, so `pip install` is all a user needs:

```toml
[project.entry-points."continuum.system_one"]
coinflip = "my_package:make_classifier"   # called as make_classifier(model)
```

Or at runtime, in a test or a single app: `register_backend("coinflip", make_classifier)`.

Then use it like any built-in backend: `SYSTEM_ONE_BACKEND=coinflip:v1`, or a
seam's own `system_one_backend=` / `backend=`. The backend is built on first
use, so a constructor that raises `SystemOneNotConfiguredError` (a missing key)
surfaces at the first decision, and each seam falls back.

A policy store that denies by default must allow the new resource,
`system_one:<egress>:coinflip:<model>` — `egress_resource(classifier)` returns it.

### 9.3 · Test it with the contract suite

```python
from continuum.system_one.testing import SystemOneContract

from my_backend import CoinFlipClassifier


class TestCoinFlip(SystemOneContract):
    def make_classifier(self):
        return CoinFlipClassifier()
```

It checks that the backend declares its capabilities and names itself, that
answer keys match the question IDs, that probabilities are in range and sum to
one, that undeclared kinds are refused, that failures are `SystemOneError`s,
and that it answers within `latency_budget_ms` (500 ms at p95 by default;
override it on the test class for a slower service).

- **The tests are async.** Run them with pytest-asyncio in auto mode
  (`asyncio_mode = "auto"` in the package's pytest config, or
  `-o asyncio_mode=auto`); without it most of them fail with *async def
  functions are not natively supported*.
- **Implement `make_failing_classifier()`.** Without it the failure test is
  skipped silently, and a backend that leaks a raw `httpx` error passes.

### 9.4 · A remote backend

Everything above applies, plus what calling a service over the network needs:

| Concern | What to do |
|---|---|
| **`egress="remote"`** | The policy resource becomes `system_one:remote:<prefix>:<model>`, so a rule denying `system_one:remote:*` to a `phi` run covers the new backend. A remote backend declaring `"local"` would send that data out; the SDK cannot detect it. |
| **Its own key** | Continuum's settings know only its built-in keys, so read yours yourself (an environment variable or a constructor argument), and raise `SystemOneNotConfiguredError` naming the variable when it is missing. Never put the key or the response body in an error or a log line. |
| **Its own timeout** | `classify()` does not time the backend; a hung service would hang the router, loop or approval gate. Set one deadline per call — `SYSTEM_ONE_TIMEOUT_SECONDS` is the setting users expect — and give every request only what remains. |
| **The right error** | no answer in time → `SystemOneTimeoutError`; connection failure or HTTP error → `SystemOneBackendError` (with `status=`); a body that is not JSON or an answer missing or malformed → `SystemOneResponseError`. |
| **Retries, if any** | Inside the deadline only; never retry a timeout — its budget is spent. |
| **Confidence and usage** | Pass the service's own figures through `raw_confidence` and `usage`. |
| **An injectable HTTP client** | So tests run against a fake service instead of the network: the latency test alone would make five paid calls per run. |

```python
import os
import time

import httpx
from continuum.config import settings
from continuum.system_one import (
    BinaryQuestion,
    SystemOneBackendError,
    SystemOneCapabilities,
    SystemOneCapabilityError,
    SystemOneNotConfiguredError,
    SystemOneRawResult,
    SystemOneResponseError,
    SystemOneTimeoutError,
)


class AcmeClassifier:
    """acme:<model> -- a hosted typed-decision service."""

    name = "acme"
    capabilities = SystemOneCapabilities(
        question_types=frozenset({"binary", "choice"}),
        egress="remote",  # state leaves the process: policy sees system_one:remote:acme:<model>
    )

    def __init__(
        self,
        model="acme-1",
        *,
        api_key=None,
        base_url="https://api.acme.example",
        timeout=None,
        client=None,
    ):
        key = api_key or os.environ.get("ACME_API_KEY")
        if not key:
            raise SystemOneNotConfiguredError(
                "The acme backend needs an API key: set ACME_API_KEY."
            )
        self.model = model
        self._key = key
        self._url = f"{base_url.rstrip('/')}/v1/decide"
        self._timeout = float(timeout or settings.system_one_timeout_seconds)
        self._client = client or httpx.AsyncClient()  # injected in tests

    async def classify(self, state, questions):
        for qid, q in questions.items():
            if q.kind not in self.capabilities.question_types:
                raise SystemOneCapabilityError(f"acme cannot answer a {q.kind} question ('{qid}')")
        payload = {
            "model": self.model,
            "state": state,
            "questions": {qid: self._wire(q) for qid, q in questions.items()},
        }
        body = await self._post(payload)
        return self._parse(body, questions)

    def _wire(self, q):
        if isinstance(q, BinaryQuestion):
            return {"type": "yes_no", "text": q.instructions}
        return {"type": "pick_one", "text": q.instructions, "options": list(q.labels)}

    async def _post(self, payload):
        deadline = time.monotonic() + self._timeout  # the SDK does not time the call: you must
        try:
            r = await self._client.post(
                self._url,
                json=payload,
                timeout=deadline - time.monotonic(),
                headers={"Authorization": f"Bearer {self._key}"},
            )
        except httpx.TimeoutException as e:
            raise SystemOneTimeoutError(
                f"acme did not answer within {self._timeout}s.", backend=self.name
            ) from e
        except httpx.HTTPError as e:
            raise SystemOneBackendError(
                f"Could not reach acme ({type(e).__name__}).", backend=self.name
            ) from e
        if not r.is_success:  # status only: never the body, never the key
            raise SystemOneBackendError(
                f"acme returned HTTP {r.status_code}.", backend=self.name, status=r.status_code
            )
        try:
            return r.json()
        except ValueError as e:
            raise SystemOneResponseError("acme returned a body that is not JSON.") from e

    def _parse(self, body, questions):
        dists, confidence = {}, {}
        try:
            for qid, q in questions.items():
                a = body["answers"][qid]
                if isinstance(q, BinaryQuestion):
                    dists[qid] = {"true": float(a["p_yes"]), "false": 1.0 - float(a["p_yes"])}
                else:
                    dists[qid] = {label: float(a["scores"][label]) for label in q.labels}
                if "confidence" in a:
                    confidence[qid] = float(a["confidence"])
        except (KeyError, TypeError, ValueError) as e:
            raise SystemOneResponseError(f"acme's answer is malformed ({type(e).__name__}).") from e
        return SystemOneRawResult(
            distributions=dists,
            raw_confidence=confidence,
            model=body.get("model", self.model),
            usage=body.get("usage", {}),
        )


def make_classifier(model):
    return AcmeClassifier(model=model)
```

Its contract test runs against a fake service, and against one that fails:

```python
import json

import httpx
from continuum.system_one.testing import SystemOneContract

from acme_backend import AcmeClassifier


def fake_acme(request):
    """Answers like the real service, without the network."""
    asked = json.loads(request.content)["questions"]
    answers = {
        qid: {"p_yes": 0.2, "confidence": 0.9}
        if q["type"] == "yes_no"
        else {"scores": {o: 1 / len(q["options"]) for o in q["options"]}, "confidence": 0.8}
        for qid, q in asked.items()
    }
    body = {"model": "acme-1", "answers": answers, "usage": {"cost": 0.0001}}
    return httpx.Response(200, json=body)


class TestAcme(SystemOneContract):
    def make_classifier(self):
        client = httpx.AsyncClient(transport=httpx.MockTransport(fake_acme))
        return AcmeClassifier(api_key="test-key", client=client)

    def make_failing_classifier(self):
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
        return AcmeClassifier(api_key="test-key", client=client)
```

If the service speaks the same typed-decision format as Jev, the SDK's
`JevOpenRouterClassifier` shows a backend that differs from Jev only in its
name, key, base URL and path. Subclassing it relies on internal names
(`_key_setting`, `_path`, `backends.typed_wire`), though, so a package outside
the SDK is safer with its own class.
