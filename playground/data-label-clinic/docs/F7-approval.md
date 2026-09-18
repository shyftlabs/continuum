# F7 — human-in-the-loop approval

**Clinic guides** — [Setup & index](TESTING_GUIDE.md) · [Labels & policy](labels-and-policy.md) · [F6 memory](F6-memory.md) · [F3 server trust](F3-server-trust.md) · [F7 approval](F7-approval.md) · [Namespacing](namespacing.md)

> Commands in this guide run from `playground/data-label-clinic/`, one level
> up from this file.

> Read [Labels & policy](labels-and-policy.md) first. This is a gate on the same
> tool call that Test 2 gates, one step later — and the scenarios below pair
> directly with [BM4](F6-memory.md), which denies the *same tool* by policy.

## What it adds over what already existed

Continuum had an approval workflow before this: `temporal/human_in_loop.py`,
`ApprovalRequest`, anti-spoofing checks. It fired only for a Temporal workflow
step of type `"approval"`. The default `AgentRunner.run()` path — the one every
quick-start uses — had no gate before any tool call.

Neither existing control covers the gap, and the difference is not a matter of
degree:


| control                             | question                                            | when                      | sees arguments?         |
| ----------------------------------- | --------------------------------------------------- | ------------------------- | ----------------------- |
| [tool-trust](F3-server-trust.md)    | is this server serving the catalogue I vetted?      | once, offline, per server | no — catalogue only     |
| [policy gate](labels-and-policy.md) | is this run allowed this resource?                  | every call, by rule       | no — `tool:{name}` only |
| **approval**                        | should *this* call, with *these* arguments, happen? | every declared call       | **yes**                 |


So neither can tell these apart:

```
transfer_funds(amount=5)            # fine
transfer_funds(amount=5_000_000)    # ask someone
```

The policy gate is binary — allow always, or deny always. It cannot express
*"allow, but ask first"*, which is the whole point for an action that is
legitimate but consequential.

## Configuration

```python
AgentConfig(
    tool_approval={"pharmacy__check_interactions"},   # fnmatch, as policy resources
    approval_handler=ask_a_human,                     # async, app-supplied
    approval_timeout=30.0,
)
```

The SDK ships **no handler and no default list of tools**. `delete_*`, `send_*`,
`pay_*` reads as a sensible default and is not one: it blocks a harmless
`send_receipt` while missing `wire_funds`, and a gate nobody configured is
assumed to cover more than it does. Same stance as `tool_data_labels` (no PII
detector) and `pre_store_filter` (no content classifier).

`CLINIC_APPROVAL` picks who answers:


| value           | handler                                                                                                                                               |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `off` (default) | none, and no tool is declared either — the gate is inert. The state a new project starts in                                                           |
| `auto`          | approves programmatically. For scripted runs that need the approved path without a browser                                                            |
| `deny`          | refuses programmatically. Shows what the model is told, without waiting for a person                                                                  |
| `ask`           | a real prompt in the web UI, blocking the turn while a reviewer answers                                                                               |
| `queue`         | refuse and resume — the turn ends at once saying *pending*, someone answers out of band, a later turn proceeds                                        |
| `temporal`      | the durable route — the turn **blocks in place** inside a Temporal activity and resumes when answered. Driven by `approval_temporal.py`, not `web.py` |


`CLINIC_APPROVAL_TIMEOUT` (default 30, or **600** under `temporal`) is how long
the gate waits. It is switchable because the limit **is** the demo — see AP4.
The default differs because the reason for 30 does not apply to `temporal`:
nothing holds an HTTP request open there, so 30 would deny a reviewer who took
half a minute — the case AP6 exists for.

### Why `check_interactions` and not `send_referral_email`

`send_referral_email` was the first choice: it is the egress tool, and "policy
denies it / a person is asked about it" pairs neatly. A live run rejected it,
in two different ways:

- **Ask for a referral email** and the model looks the patient up first. That
taints the run `phi`, so `phi-no-exfiltration-tools` denies the call *before*
approval is consulted. All three modes produce identical output.
- **Forbid the lookup** and the model declines to send mail at all. `tools: []`
in all three modes.

Both were observed before anything was committed. It is the same trap
[BM4](F6-memory.md) documents — *a gate nothing reaches demonstrates nothing* —
which is why `check_interactions` carries the EXTERNAL rule there. It carries
the approval gate here for the same reason, and the pairing survives intact:
**BM4 shows the policy denying** `check_interactions` **to an EXTERNAL run; AP1–AP4
show a person being asked about it on a clean one.** Same action, two postures,
a rule deciding versus a human deciding.

---



## Before any of these — three terminals

The gated tool is `check_interactions`, which lives on the **pharmacy** server.
Both servers must be up before `web.py`, and not just so the tool exists — the
agent refuses to initialise without them.

```bash
python server.py            # :8911 — clinic MCP tools
python pharmacy_server.py   # :8912 — pharmacy MCP tools, incl. check_interactions
python web.py               # :8910 — the UI, with CLINIC_APPROVAL set
```

AP6 replaces the third terminal with `approval_temporal.py` but still needs the
first two.

---



## AP1 — a declared tool asks, and proceeds when approved

1. Start `web.py` with `CLINIC_APPROVAL=auto`, both MCP servers already up.
2. Send **"Check for interactions between metformin and lisinopril."**

```
tools: ['pharmacy__check_interactions']
reply: There are no clinically significant interactions between metformin and
       lisinopril. They are commonly co-prescribed.
```

Identical to `CLINIC_APPROVAL=off`, and that is the point: an approved call is
indistinguishable from an ungated one. The gate is not a speed bump.

- **What it proves:** the gate fires on a declared tool and does not change the
outcome when the answer is yes. Compare against `off` in the same session —
if the two differ, the gate is doing something it should not.



## AP2 — a refusal is relayed, not raised

3. Restart with `CLINIC_APPROVAL=deny` and send the same message.

```
gate:  ⏸ TOOL — not approved: APPROVAL DENIED: 'pharmacy__check_interactions'
       was not approved by auto (CLINIC_APPROVAL=deny). Refused by the scripted
       reviewer.
reply: The action to check for interactions between metformin and lisinopril was
       not performed as it requires approval and was denied by the system.
```

- **What it proves:** a refusal becomes a **tool result**, so the model tells the
user and the turn ends normally. The run does not crash over a decision the
system made on purpose — the same ergonomics as `POLICY DENIED` in Test 2.
- **Note the ⏸ rather than 🛡️.** The panel separates the two because *a rule
refused* and *a person refused* are different events, and which one happened
is the part worth seeing. A refusal that did not appear in the panel would be
indistinguishable from the tool quietly not running.



## AP3 — the prompt carries the arguments

4. Restart with `CLINIC_APPROVAL=ask` and send the same message. The chat pauses
  and a prompt appears in the transcript:

```
⏸ APPROVAL NEEDED
pharmacy__check_interactions
{ "medications": [ "metformin", "lisinopril" ] }
[approve] [deny]
```

5. Click **approve**. The turn resumes and the tool runs.

- **What it proves:** the handler receives the **arguments**, which is the
capability this finding adds. A reviewer shown only a tool name is approving
the name. Neither tool-trust nor the policy gate can see this payload, so
neither could have told a routine lookup from a consequential one.
- The run's taint is shown alongside, when there is any. *"This run has read the
public web"* is exactly what a reviewer wants to know before approving an
outbound action, and it is not visible from the tool name or the arguments.

> **How the prompt reaches the browser, and why it looks like this.**
> `POST /chat` is already in flight and blocked inside the tool executor when the
> handler runs, so the decision cannot come back on that connection. The page
> opens a **second** request: it polls `/approval/pending` while the chat request
> is still open, and posts the answer to `/approval/decide`, which resolves the
> future the handler is parked on (`approval_ui.py`).
>
> That is the shape of every blocking-approval UI over HTTP — and also its
> limit. The chat request stays open the whole time.



## AP4 — nobody answers, and it fails closed

6. Restart with `CLINIC_APPROVAL=ask CLINIC_APPROVAL_TIMEOUT=3`, send the same
  message, and **do not click anything**.

```
gate:  ⏸ TOOL — not approved: APPROVAL DENIED: 'pharmacy__check_interactions'
       was not approved. No reviewer responded within 3.0s (timed out).
reply: The request ... was not completed because it requires approval.
```

Clicking after the timeout reports *"too late — the approval already timed out
and was denied"*, and the prompt clears itself from the panel rather than
sitting there claiming to be live.

- **What it proves:** the gate fails **closed** on every path — a handler that
raises, one that never answers, one that returns something other than a
decision, and a tool declared for approval with no handler wired. That last
one is the configuration most easily mistaken for protection, so it refuses
rather than quietly proceeding.
- **Why the timeout is short by default.** A blocked run holds the HTTP request
open while the reviewer thinks, and browsers, proxies and serverless platforms
give up long before a person does. 30s sits inside ordinary limits. Set
`CLINIC_APPROVAL_TIMEOUT=3` and walk away to see what a reviewer who takes two
minutes behind a proxy will actually experience.



## AP5 — refuse and resume, for a reviewer who is not watching

`ask` holds the HTTP request open while someone decides, which only works if
they are looking at the screen. `queue` is the other shape, and the one most
asynchronous deployments actually need.

7. Restart with `CLINIC_APPROVAL=queue` and send **"Check for interactions
  between metformin and lisinopril."** The turn ends immediately:

```
gate:  ⏳ TOOL — awaiting approval: APPROVAL PENDING:
       'pharmacy__check_interactions' has been sent for approval. Queued for a
       reviewer. Ask again once it has been answered.
reply: The request to check interactions between metformin and lisinopril is
       awaiting approval. Please ask again once it has been reviewed.
```

8. Answer it out of band — no turn is waiting on this:

```bash
curl -s localhost:8910/approval/queued
curl -sX POST localhost:8910/approval/answer -H 'Content-Type: application/json' \
     -d '{"key":"<key from above>","approved":true,"reviewer":"tom"}'
```

9. Ask the same question again. It proceeds:

```
tools: ['pharmacy__check_interactions']
reply: There are no clinically significant interactions between metformin and
       lisinopril. They are commonly co-prescribed.
```

- **What it proves:** a third outcome exists. Without it a deferred call and a
refused one are the same value (`approved=False`), so the model says APPROVAL
DENIED for something that is merely waiting — and a user told their request
was *refused* does not go looking for an approver. The panel separates them
too: **⏳** for pending, **⏸** for declined. One is resumable and the other is
not, which is the whole reason the state exists.
- **Only a handler may defer.** A timeout or a crash is a refusal, never a
deferral: nobody answered and nothing is queued, so reporting it as pending
would promise a resumption that nothing is going to deliver.
- **The resume half is the application's.** The SDK remembers nothing between
runs — a cached approval could authorise an execution the reviewer never saw —
so `approval_ui.py` supplies the store and its rules. It keys on
*(tool, arguments)*, because the second turn is a different run asking the
same question and there is no request id to carry over. An answer authorises
**one** execution and is then forgotten; asking a third time queues again. A
standing permit would be exactly the hazard the SDK declines to build in.



## AP6 — a reviewer who answers in an hour (Temporal)

`ask` holds an HTTP request open, so the reviewer has to be watching. `queue`
lets them answer whenever, but the turn ends and the user asks again, redoing
whatever came before the gate. The durable route is the third shape: the turn
**blocks in place** and resumes when answered.

It needs a Temporal workflow, which `web.py` does not run — so this mode is
driven by `approval_temporal.py` rather than the browser, and the reviewer
answers in Temporal's own UI. Everything else is the clinic's own agent, MCP
servers and policy store, so the call being gated is the same one AP1–AP5 gate.
The handler is the SDK's `temporal_tool_approval()`, which finds its own
workflow at call time.

### Why this one leaves the clinic's UI

`web.py` runs the agent in-process — `POST /chat` calls `_agent.chat()`, which
calls `runner.run()`. There is no workflow anywhere in that path, and
`temporal_tool_approval()` resolves its workflow from `activity.info()`, so
inside an HTTP handler it finds no activity and defers every request rather than
letting the call through unreviewed. Something has to start a workflow and run a
worker; that something is the driver.

This is a choice, not a limit. The clinic already *has* a browser approval UI —
`CLINIC_APPROVAL=ask`, AP1–AP4, with Approve and Deny buttons in the chat. What
it cannot do is survive, because the turn is held inside an open HTTP request.
AP6's claim is not a nicer reviewer view; it is **where the wait lives**:


| mode       | reviewer sees                | the wait is held in                                   |
| ---------- | ---------------------------- | ----------------------------------------------------- |
| `ask`      | the clinic chat, in-line     | an open HTTP request — browser and proxy limits apply |
| `queue`    | the clinic chat, out of band | nothing at all; the turn already ended                |
| `temporal` | Temporal UI at :8233         | durable workflow state                                |


Wiring the clinic's own UI to the workflow is perfectly possible, and the cost
is worth naming because it is not where you would guess. The approval endpoints
are trivial to repoint: `/approval/pending` becomes a
`handle.query("get_pending_approvals")` and `/approval/decide` a
`handle.signal("submit_approval", …)`, a few dozen lines including tracking a
workflow id per conversation. The prompt card in the page already exists.

The expense is `/chat`. `ClinicAgent.chat()` returns `response`, `taint`,
`model_used`, `gate_events` and `tools_called` — those last four *are* the
glassbox panels on the right of the page, and they are the reason this
playground exists. Route the turn through Temporal and `run_agent_activity`
hands back an `AgentActivityResult`: `content`, `status`, `usage`,
`agents_used`. No taint, no gate events, no tool list. Every panel goes blank
unless you either change an SDK type on the shared Temporal path or carry the
glassbox data through `metadata` as a second result shape. Streaming goes too —
a workflow returns a result, it does not yield tokens, so the page's `stream`
checkbox would need its own channel.

So the demo borrows Temporal's UI instead. Not because the clinic's UI could not
show a durable approval, but because making it do so would cost the clinic the
thing it is actually demonstrating.

10. Start Temporal and both MCP servers:

```bash
# from the repo root. temporal-ui sits behind the `full` profile, so naming it
# explicitly is what starts it; the UI then serves on 8233, not its container's 8080.
docker compose up -d temporal postgres-temporal temporal-ui

# from playground/data-label-clinic, one per terminal
python server.py
python pharmacy_server.py
```

11. Run both scripted paths. No browser, no reviewer — these prove the round
  trip works before you try answering one by hand:

```bash
python approval_temporal.py --auto approve
python approval_temporal.py --auto deny
```

Measured, against Temporal 1.29.3 with a real worker and a real LLM call:

```
connected to temporal at localhost:7233
gated tool: pharmacy__check_interactions
worker up on clinic-f7-approval
workflow: f7-04301c39

PROMPT
  request_id: tool-04301c39ee7c
  tool:       pharmacy__check_interactions
  arguments:  {"medications": ["metformin", "lisinopril"]}

APPROVED by tom

status:  completed
answer:  There are no clinically significant interactions between metformin and
         lisinopril. They are commonly co-prescribed.
decision: tool-04301c39ee7c approved by tom
```

and the refusal, same setup:

```
REJECTED by tom

status:  completed
answer:  The action to check for interactions between metformin and lisinopril
         was not approved and therefore was not performed. It requires approval
         to proceed.
decision: tool-3e8d15260cd8 rejected by tom
```

12. Now answer one **yourself**, which is the actual scenario. Run it with no
  `--auto`:

```bash
python approval_temporal.py
```

It prints the prompt and stops — the turn is blocked inside the activity,
heartbeating, holding no connection open. Two ways to answer it.

**From the Temporal UI** at [localhost:8233](http://localhost:8233). Open the
workflow it printed, then **More Actions → Send a Signal**:


| field       | value                                                                                                                                               |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Signal name | `submit_approval` — pick it from the dropdown; the workflow advertises it, along with `request_tool_approval`, `inject_input` and `cancel_workflow` |
| Data        | the JSON below, with the `request_id` the driver printed                                                                                            |
| Encoding    | `json/plain` (the default)                                                                                                                          |


```json
{"request_id": "tool-1bf5610c6886", "decision": "approved", "decided_by": "tom"}
```

A plain JSON object is enough: the connection uses Temporal's pydantic data
converter, so the payload is validated into `ApprovalDecision` on the way in.
`decided_by` is what lands in the audit trail — put a real name there and you
will see it come back as `decision: … approved by <name>`. For a refusal, change
`"decision"` to `"rejected"`; an optional `"reason"` is relayed to the model.

> `request_id` **is the** `tool-…` **id, not the workflow id.** They sit next to
> each other on that page and the workflow id is the one in the heading, so it
> is the easy thing to paste. A workflow id there is not an error — it
> deserializes perfectly well and is simply a decision for a request that does
> not exist, so the signal is accepted, nothing resolves, and the run stays
> blocked with no complaint. Copy the `request_id:` line from the driver's
> output.

> **Make the window tall enough first.** The dialog is longer than a short
> browser window, and the UI's footer bar covers the Submit button when it is
> below the fold — the click registers on nothing, the dialog stays open, and
> no signal is sent. There is no error, and the workflow simply stays
> `Running`. This cost four attempts to spot. If a submit seems to do nothing,
> maximise the window and try again, then check **Event History** for a second
> `WorkflowExecutionSignaled`.

> **Submitting an empty Data field used to park the workflow.** The signal
> arrived with no payload, `submit_approval` raised `TypeError`, and a signal
> handler that raises fails the workflow *activation* — which Temporal retries
> forever. The run could then never be answered, including by a correct signal
> sent afterwards. It is now dropped with a warning naming the cause, so a
> mistyped form costs you one log line rather than the run.

**Or from another shell**, using the `workflow` and `request_id` it printed:

```python
import asyncio, os
os.environ.setdefault("CLINIC_APPROVAL", "temporal")
from continuum.temporal import get_temporal_client
from continuum.temporal.types import ApprovalDecision

async def main():
    c = get_temporal_client()
    await c.connect("localhost:7233")
    h = c.raw_client.get_workflow_handle("f7-d281aa19")      # the printed workflow
    await h.signal("submit_approval", ApprovalDecision(
        request_id="tool-475029b4b197",                      # the printed request_id
        decision="approved", decided_by="tom"))

asyncio.run(main())
```

Measured: the run blocked at 13:20:53, sat there while the reviewer did
something else, and on the signal resumed and completed — **one** workflow,
**one** LLM call, no second ask. That gap was about four minutes, comfortably
past the 30s `ask` is stuck with.

```
PROMPT
  request_id: tool-475029b4b197
  arguments:  {"medications": ["metformin", "lisinopril"]}
waiting for a reviewer. Answer it with submit_approval —

status:  completed
answer:  There are no clinically significant interactions between metformin and
         lisinopril. They are commonly co-prescribed.
decision: tool-475029b4b197 approved by tom
```

Measured through the UI, same mechanism, a different reviewer name so the trail
is unambiguous about which route answered it:

```
decision: tool-1bf5610c6886 approved by tom-via-ui
```

and in the workflow's Event History, the second signal is the answer:

```
 6  17:44:19Z  WorkflowExecutionSignaled     ← request_tool_approval, from the activity
10  17:49:17Z  WorkflowExecutionSignaled     ← submit_approval, from the UI
19  17:49:21Z  WorkflowExecutionCompleted
```

Five minutes blocked, then four seconds to finish once answered. That gap is the
claim: nothing was recomputed, and `decided_by` typed into a browser form came
back through `ApprovalDecision` untouched.

While it is blocked you can read the request off the workflow yourself — this is
what a reviewer's UI calls:

```python
await h.query("get_pending_approvals")
# [{'request_id': 'tool-475029b4b197', 'workflow_id': 'f7-d281aa19',
#   'description': 'pharmacy__check_interactions',
#   'context': '{"medications": ["metformin", "lisinopril"]}', 'approvers': []}]
```

- **What it proves:** the turn resumed **in place**. The work before the gate was
not repeated and nobody had to ask again — the one thing this adds over AP5.
The prompt carried the arguments, as everywhere else.
- **One signal, one reviewer.** `submit_approval` serves both a planned approval
step and an ad-hoc tool approval, with the same allow-list check via
`is_authorized`. A tool approval is not a second, weaker door into the same
workflow.
- **Every failure defers, never denies.** An unreachable workflow, a query that
fails mid-wait, or a `request_id` the workflow does not recognise all return
*deferred*. Temporal being down is not a reviewer saying no, and `unknown` is
a distinct status from `pending` precisely so silence cannot be read as
permission.

> **The gotcha that cost four attempts.** `temporal_tool_approval()` resolves
> its handle through the **global** Temporal client, and a worker connects its
> **own** — they are different objects. So a setup that looks entirely correct,
> with a connected worker happily running activities, defers every approval with
> *"Not connected to Temporal server"*. It fails safe, which is the good half,
> but it reads as a network fault rather than a missing line:
>
> ```python
> await get_temporal_client().connect(host)
> ```
>
> The deferral message now names this, because it is the only part an operator
> sees. `approval_temporal.py` makes the call, so you only hit this writing your
> own driver.

> **A second one, if you write your own driver.** Do not call
> `worker.register_workflow(AgentWorkflow)` or
> `worker.register_activity(run_agent_activity)`. `WorkerManager.start` already
> registers both, and registering them again is a hard failure at worker
> startup — `ValueError: More than one activity named run_agent_activity` —
> not a duplicate that gets ignored.

> **And print with line buffering.** The one output you are waiting for is
> produced *while the turn is blocked*. Pipe a driver's stdout anywhere and
> Python block-buffers it, so a correctly blocked run looks exactly like a hung
> one. `approval_temporal.py` calls `sys.stdout.reconfigure(line_buffering=True)`
> for this reason.

---



## What this layer does and doesn't cover

- **Covers:** the gate firing on a declared tool, an approval proceeding, a
refusal relayed as a tool result, a real prompt carrying the arguments,
failing closed on a timeout, the refuse-and-resume round trip, and a durable
wait through Temporal — live, with the ungated `off` mode as the control for
each.
- **Covers an hours-long wait, through Temporal** — see AP6. This paragraph
used to say the opposite, and the history is worth keeping because it explains
the shape. `HumanInLoopManager` is the decision-*submission* side — `approve`,
`reject`, `submit_decision` — the API a reviewer's UI calls. The waiting lived
inside the workflow, whose `_run_approval_step` appends to
`_pending_approvals` and blocks on a signal, and nothing outside could
register a request. An approval handler runs wherever the tool call runs,
which under Temporal is inside an **activity**, where workflow APIs are
unavailable by design. Those two facts together made the route impossible, and
a test asserted them.
That test failed the moment `request_tool_approval` (a signal) and
`get_approval_decision` (a query) were added, which is what a tripwire is for.
- **Doesn't cover: surviving a worker restart.** The remaining limit, and a real
one. The whole agent turn is ONE activity — `run_agent_activity` calls
`runner.run()` — so the workflow cannot pause *between* the agent's own steps.
The activity blocks in place, which is why work done before the gate is not
repeated and nobody has to ask again; but a retried activity starts from the
beginning. A reviewer at lunch is fine. A reviewer who outlasts a deploy is
not, and for them `CLINIC_APPROVAL=queue` (AP5) is the honest answer, because
it holds nothing open at all.
- **Approvals are serialised, not batched.** Tool calls run through
`asyncio.gather`, so without a lock two handlers fire at once — two prompts
racing for one terminal. Serialising costs wall-clock when several calls in
one turn need approval, and it means a reviewer sees one call at a time rather
than *"this turn wants to do these three things"*.
- **Approvals are not remembered.** A retried run asks again. Idempotency sounds
obviously right and is a loaded gun: a cached "approve" authorises a second
execution the reviewer never saw, and whether a retry is the same action or a
new one depends on the tool — transferring money and sending a reminder want
opposite answers. `run_id` and `arguments` are on the request so an app can
build the policy it needs; the SDK does not pick one, because half-built
idempotency looks like protection while quietly authorising repeats. If you
want it remembered, do it in your app — see below.



### If you want approvals remembered

Do it in your app, not in the SDK. An approval handler is just
`async (request) -> decision`, so wrap yours in one that looks up a store first
and writes the answer back after.

```python
approval_handler = remembering(my_handler, key=..., store=redis)
```

You choose the key — the arguments, a business id, or nothing at all — because
only you know whether running your tool twice is safe. Two rules: never cache a
`deferred` (nobody answered yet), and delete a cached answer once it is used, so
one approval authorises one execution rather than becoming a standing permit.
`queue_approval_handler` in `approval_ui.py` is a working example.

---



## Where each scenario hits the SDK


| Scenario | SDK code path exercised                                                                                                                                                                                   |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| AP1      | the gate in `ToolExecutor.execute_tool_call` — after the policy check, before argument injection, outside the tool-execution `asyncio.wait_for`                                                           |
| AP2      | `ToolApprovalDeniedError` (a `PolicyDeniedError`) rendered by `ToolExecutor._process_tool_results` as `APPROVAL DENIED`                                                                                   |
| AP3      | `ToolApprovalRequest` carrying `arguments` and `data_labels`; `request_approval` serialising on a per-event-loop lock                                                                                     |
| AP4      | `request_approval`'s fail-closed paths — timeout, raise, wrong return type, and no handler wired                                                                                                          |
| AP5      | `ToolApprovalDecision(deferred=True)` → `APPROVAL PENDING` rather than `APPROVAL DENIED`; the resume store is the app's (`approval_ui.py`)                                                                |
| AP6      | `temporal_tool_approval` resolving its own handle from `activity.info()`; the workflow's `request_tool_approval` signal and `get_approval_decision` query; `submit_approval` shared with the planned step |
| all      | `build_approval_settings` reading `AgentConfig`, passed at both `ToolService` call sites                                                                                                                  |


