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

```toml
[project.entry-points."continuum.system_one"]
mybackend = "my_package:make_classifier"   # called as make_classifier(model)
```

`mybackend:<model>` then resolves with no change to the SDK.
`register_backend(prefix, factory)` does the same at runtime. A backend
implements `ISystemOneClassifier` (`continuum.protocols`) and should pass the
contract suite:

```python
from continuum.system_one.testing import SystemOneContract

class TestMyBackend(SystemOneContract):
    def make_classifier(self):
        return MyClassifier(model="my-model")
```

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
