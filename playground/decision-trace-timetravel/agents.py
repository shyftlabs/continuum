"""
The month-end close (local/glassbox) expressed across all 9 multi-agent patterns
(+ handoff), recreated to local/glassbox's quality bar.

Universal lever: the **materiality threshold**. Every topology's verdict is
computed by the deterministic ``compute_consolidation(threshold)`` tool, so
lowering the threshold past $2M deterministically flips CONTROL_ISSUE → CLEAN in
every mode — regardless of LLM phrasing. Each builder places that threshold at a
*forkable* point so a rewind both flips the result and (where the topology has an
upstream) replays earlier work from cache.

  sequential  ingest→reconcile→variance→intercompany→consolidate→report  (local/glassbox, reused)
  supervised  intake→variance→decide, each step supervisor-scored
  planning    plan → {facts, variance, decide}
  loop        close-officer ↻ refines until DECISION
  reflection  drafter → critique (does suspense balance?) → revise
  parallel    materiality-decider ‖ intercompany-analyst → relay
  scatter     close facets ‖ → gather
  router      route to strict($1M) vs lenient($5M) close
  debate      "materially fine" vs "D1 is material" → judge computes
  handoff     triage → close-officer → controller (final, computes)

Lever placement per mode (where a fork edits the threshold):
  sequential/supervised/planning : the Variance stage input (an upstream stage
      emits MATERIALITY_THRESHOLD_USD=<n>) → fork mid-pipeline = flip + savings.
  loop/reflection                : the iteration/attempt input carries the line.
  parallel/scatter               : the materiality branch input → fork one branch,
      re-merge cached siblings.
  router                         : re-route strict/lenient.
  debate                         : the judge's input threshold.
  handoff                        : the get_materiality_policy tool result
      (single-agent resume honors set_tool_result).
"""

from __future__ import annotations

from typing import Any

from continuum import AgentConfig, AgentMemoryConfig, BaseAgent, Handoff
from continuum.agent.config import ParallelConfig, RouterConfig
from continuum.agent.types import MergeStrategy, Route, TerminationType
from continuum.agent.workflow import (
    create_planner_agent,
    create_reflection_agent,
    create_scatter_agent,
    create_supervised_agent,
)
from continuum.agent.workflow.debate import DebateAgent
from continuum.agent.workflow.loop import create_loop_agent
from continuum.agent.workflow.parallel import ParallelAgent
from continuum.agent.workflow.router import RouterAgent
from continuum.agent.workflow.sequential import SequentialAgent

_MEM = AgentMemoryConfig(search_memories=False, store_memories=False)
_CFG = AgentConfig(log_to_session=False, session_history_turns=0)

# The decision tail every deciding agent must emit, so the UI/_outcome detects it.
_DECIDE_TAIL = (
    "End your reply with EXACTLY these two lines (matching the tool's status field):\n"
    "STATUS=<CLEAN or CONTROL_ISSUE>\n"
    "DECISION: CLEAN   (or)   DECISION: CONTROL_ISSUE"
)

# How every decider must obtain the threshold — deterministically, never by asking.
_GET_THRESHOLD = (
    "Obtain the materiality threshold first: if your input contains a line "
    "'MATERIALITY_THRESHOLD_USD=<n>', use that integer n; OTHERWISE you MUST call "
    "get_materiality_policy and use its materiality_threshold_usd. Never ask the user "
    "for the threshold, never request it, and never stop without computing — always "
    "proceed to the tools below.\n"
)


def _tools(tools: list[dict[str, Any]], names: set[str]) -> list[dict[str, Any]]:
    return [t for t in tools if t.get("function", {}).get("name") in names]


def _agent(name, instructions, tools, te, model, tool_names=None):
    kw: dict[str, Any] = {}
    if tool_names:
        kw["tools"] = _tools(tools, tool_names)
        kw["tool_executor"] = te
    return BaseAgent(
        name=name,
        instructions=instructions,
        model=model,
        memory_config=_MEM,
        config=_CFG,
        **kw,
    )


# --------------------------------------------------------------------------- #
# Shared close stages
# --------------------------------------------------------------------------- #
def _full_close_stages(p, tools, te, model) -> list[BaseAgent]:
    """The faithful 6-stage close (local/glassbox), name-prefixed by mode ``p``."""
    return [
        _agent(
            f"{p}-ingest",
            "You are the ingestion agent. Call get_close_data and confirm the entities and "
            "their combined totals in one short paragraph. Never alter figures.",
            tools,
            te,
            model,
            {"get_close_data"},
        ),
        _agent(
            f"{p}-reconcile",
            "You are the reconciliation agent. Call get_close_data for the discrepancies and "
            "get_materiality_policy for the threshold. Give a one-line risk note per "
            "discrepancy. Do NOT change amounts or ids.\nYour reply MUST end with EXACTLY:\n"
            "MATERIALITY_THRESHOLD_USD=<the materiality_threshold_usd integer from the policy>",
            tools,
            te,
            model,
            {"get_close_data", "get_materiality_policy"},
        ),
        _agent(
            f"{p}-variance",
            "You are the variance & materiality agent. Your input contains a line "
            "'MATERIALITY_THRESHOLD_USD=<n>'. Read that integer as the threshold.\n"
            "Call assess_materiality with threshold_usd=that integer. Narrate which "
            "discrepancies are material, calling out any affects_balance item waived as "
            "immaterial.\nYour reply MUST end with EXACTLY:\nTHRESHOLD_USED=<that integer>",
            tools,
            te,
            model,
            {"assess_materiality"},
        ),
        _agent(
            f"{p}-intercompany",
            "You are the intercompany eliminations agent. Call get_intercompany and confirm "
            "which transactions are eliminated on consolidation, in one short paragraph.",
            tools,
            te,
            model,
            {"get_intercompany"},
        ),
        _agent(
            f"{p}-consolidate",
            "You are the consolidation agent. Find the line 'THRESHOLD_USED=<n>' from the "
            "variance step in the prior context and read the integer n.\nCall "
            "compute_consolidation with threshold_usd=n. Narrate whether the suspense account "
            "nets to zero (do not recompute).\n" + _DECIDE_TAIL,
            tools,
            te,
            model,
            {"compute_consolidation"},
        ),
        _agent(
            f"{p}-report",
            "You are the reporting agent. Your input contains a 'STATUS=<CLEAN|CONTROL_ISSUE>' "
            "line from the consolidation step — that is the authoritative verdict. Do NOT "
            "re-judge it: a material discrepancy that was BOOKED and nets the suspense to zero "
            "is CLEAN, not a control issue.\nWrite a 2-3 sentence controller-to-CFO summary, "
            "then end with EXACTLY one line whose value MUST equal that STATUS:\n"
            "DECISION: CLEAN   (or)   DECISION: CONTROL_ISSUE",
            tools,
            te,
            model,
        ),
    ]


def _compact_close_stages(p, tools, te, model) -> list[BaseAgent]:
    """A 3-stage close core (intake emits the threshold → variance → decide), so a
    fork of the Variance stage replays intake and flips the verdict."""
    return [
        _agent(
            f"{p}-intake",
            "You are the close intake agent. Call get_close_data (discrepancies) and "
            "get_materiality_policy (threshold). Restate the discrepancies in one short "
            "paragraph.\nYour reply MUST end with EXACTLY:\n"
            "MATERIALITY_THRESHOLD_USD=<the materiality_threshold_usd integer>",
            tools,
            te,
            model,
            {"get_close_data", "get_materiality_policy"},
        ),
        _agent(
            f"{p}-variance",
            "You are the variance agent. Read 'MATERIALITY_THRESHOLD_USD=<n>' from your input. "
            "Call assess_materiality with threshold_usd=n. Narrate the material vs waived "
            "items.\nYour reply MUST end with EXACTLY:\nTHRESHOLD_USED=<n>",
            tools,
            te,
            model,
            {"assess_materiality"},
        ),
        _agent(
            f"{p}-decide",
            "You are the consolidation & decision agent. Read 'THRESHOLD_USED=<n>' from the "
            "prior context. Call compute_consolidation with threshold_usd=n and narrate "
            "whether the suspense nets to zero.\n" + _DECIDE_TAIL,
            tools,
            te,
            model,
            {"compute_consolidation"},
        ),
    ]


# --------------------------------------------------------------------------- #
# 1 · Sequential  (local/glassbox, reused verbatim)
# --------------------------------------------------------------------------- #
def build_sequential(tools, te, model):
    stages = _full_close_stages("seq", tools, te, model)
    return SequentialAgent(
        name="close-sequential", agents=stages, sequential_config=__seq_cfg()
    ), stages


def __seq_cfg():
    from continuum.agent.config import SequentialConfig

    return SequentialConfig(pipeline_context_max_chars=None)


# --------------------------------------------------------------------------- #
# 2 · Supervised  (compact close, each step supervisor-scored)
# --------------------------------------------------------------------------- #
def build_supervised(tools, te, model):
    stages = _compact_close_stages("sup", tools, te, model)
    sup = create_supervised_agent(
        "close-supervised", stages, quality_threshold=0.6, max_retries=1, supervisor_model=model
    )
    return sup, stages


# --------------------------------------------------------------------------- #
# 3 · Planning  (plan the close, execute the steps)
# --------------------------------------------------------------------------- #
def build_planning(tools, te, model):
    stages = _compact_close_stages("plan", tools, te, model)
    planner = create_planner_agent(
        "close-planner",
        agents=stages,
        instructions="Decompose the month-end close into ordered steps: intake the close data "
        "and threshold, assess materiality, then consolidate and decide. Assign each step to the "
        "appropriate agent by name.",
        max_steps=5,
        planning_model=model,
    )
    return planner, stages


# --------------------------------------------------------------------------- #
# 4 · Loop  (one close-officer iterates until it emits a DECISION)
# --------------------------------------------------------------------------- #
def build_loop(tools, te, model):
    officer = _agent(
        "loop-close-officer",
        "You run a month-end close, refining your judgement each turn.\n"
        + _GET_THRESHOLD
        + "Call assess_materiality(threshold_usd=n) then compute_consolidation(threshold_usd=n). "
        "Narrate, then end with EXACTLY:\n" + _DECIDE_TAIL,
        tools,
        te,
        model,
        {"get_materiality_policy", "assess_materiality", "compute_consolidation"},
    )
    loop = create_loop_agent(
        "close-loop",
        officer,
        termination_type=TerminationType.OUTPUT_MATCH,
        termination_pattern=r"DECISION:\s*(CLEAN|CONTROL_ISSUE)",
        max_iterations=3,
    )
    return loop, [officer]


# --------------------------------------------------------------------------- #
# 5 · Reflection  (draft the close, critique whether it balances, revise)
# --------------------------------------------------------------------------- #
def build_reflection(tools, te, model):
    drafter = _agent(
        "refl-drafter",
        "Draft the month-end close verdict.\n"
        + _GET_THRESHOLD
        + "Call assess_materiality(threshold_usd=n) then compute_consolidation(threshold_usd=n), "
        "then end with EXACTLY:\n" + _DECIDE_TAIL,
        tools,
        te,
        model,
        {"get_materiality_policy", "assess_materiality", "compute_consolidation"},
    )
    refl = create_reflection_agent(
        "close-reflection",
        drafter,
        # ReflectionAgent stops when the critique reply starts with 'PASS' (else it
        # forces a revise). Speak that protocol explicitly, or a correct draft is
        # needlessly revised — and the revise can drop the STATUS=/DECISION: verdict.
        critique_prompt=(
            "You are reviewing a month-end close verdict. Reply with EXACTLY the word "
            "'PASS' (nothing before it) if ALL hold: the reported suspense balance "
            "matches compute_consolidation, the CLEAN/CONTROL_ISSUE verdict correctly "
            "reflects whether suspense nets to zero, and the reply ends with the "
            "STATUS= and DECISION: lines. Otherwise reply 'NEEDS IMPROVEMENT: <what to "
            "fix>'. If you ask for changes, the reviser MUST re-call assess_materiality "
            "and compute_consolidation and end with the STATUS=/DECISION: lines."
        ),
        max_reflections=1,
        reflection_model=model,
    )
    return refl, [drafter]


# --------------------------------------------------------------------------- #
# 6 · Parallel  (materiality decider ‖ intercompany analyst → relay)
# --------------------------------------------------------------------------- #
def build_parallel(tools, te, model):
    decider = _agent(
        "par-materiality",
        "You own the close verdict.\n"
        + _GET_THRESHOLD
        + "Call assess_materiality(threshold_usd=n) then compute_consolidation(threshold_usd=n). "
        "End with EXACTLY:\n" + _DECIDE_TAIL,
        tools,
        te,
        model,
        {"get_materiality_policy", "assess_materiality", "compute_consolidation"},
    )
    ic = _agent(
        "par-intercompany",
        "Call get_intercompany and summarize the intercompany eliminations in one paragraph "
        "(context only; not the verdict).",
        tools,
        te,
        model,
        {"get_intercompany"},
    )
    par = ParallelAgent(
        name="close-parallel",
        agents=[decider, ic],
        model=model,
        parallel_config=ParallelConfig(
            merge_strategy=MergeStrategy.LLM_SUMMARIZE,
            summary_prompt="The materiality decider's analysis ends with a 'STATUS=<...>' line and "
            "a 'DECISION: ...' line. Copy THOSE TWO LINES EXACTLY as your final two lines — do not "
            "invent, change, or guess the verdict. Add one sentence of intercompany context before "
            "them.",
            summary_model=model,
        ),
        memory_config=_MEM,
        config=_CFG,
    )
    return par, [decider, ic]


# --------------------------------------------------------------------------- #
# 7 · Scatter  (close facets in parallel → gather)
# --------------------------------------------------------------------------- #
def build_scatter(tools, te, model):
    decider = _agent(
        "scat-materiality",
        "You own the close verdict.\n"
        + _GET_THRESHOLD
        + "Call assess_materiality(threshold_usd=n) then compute_consolidation(threshold_usd=n). "
        "End with EXACTLY:\n" + _DECIDE_TAIL,
        tools,
        te,
        model,
        {"get_materiality_policy", "assess_materiality", "compute_consolidation"},
    )
    ic = _agent(
        "scat-intercompany",
        "Call get_intercompany and report the eliminations in one paragraph (context).",
        tools,
        te,
        model,
        {"get_intercompany"},
    )
    sc = create_scatter_agent(
        "close-scatter",
        [decider, ic],
        merge_strategy=MergeStrategy.LLM_SUMMARIZE,
        split_model=model,
    )
    sc.model = model  # the gather step calls the LLM directly and needs a model
    # Preserve the verdict through the gather (mirrors build_parallel): the
    # materiality branch ends with 'STATUS=<...>' / 'DECISION: ...' lines — the
    # merge MUST copy them verbatim, or the final answer carries no verdict.
    sc.scatter_config.summary_model = model
    sc.scatter_config.summary_prompt = (
        "One branch (the materiality decider) ends with a 'STATUS=<...>' line and a "
        "'DECISION: ...' line. Copy THOSE TWO LINES EXACTLY as your final two lines — do "
        "not invent, change, or guess the verdict. Add one sentence of intercompany "
        "context before them."
    )
    return sc, [decider, ic]


# --------------------------------------------------------------------------- #
# 8 · Router  (route to a strict or lenient close — deterministic per route)
# --------------------------------------------------------------------------- #
def build_router(tools, te, model):
    strict = _agent(
        "strict-close",
        "You run a STRICT close at a $1,000,000 materiality threshold. Call "
        "compute_consolidation(threshold_usd=1000000) and report. End with EXACTLY:\n"
        + _DECIDE_TAIL,
        tools,
        te,
        model,
        {"compute_consolidation"},
    )
    lenient = _agent(
        "lenient-close",
        "You run a LENIENT close at the firm's $5,000,000 materiality threshold. Call "
        "compute_consolidation(threshold_usd=5000000) and report. End with EXACTLY:\n"
        + _DECIDE_TAIL,
        tools,
        te,
        model,
        {"compute_consolidation"},
    )
    router = RouterAgent(
        name="close-router",
        routes=[
            Route(
                agent_name="lenient-close",
                description="standard firm policy close",
                condition="lenient",
            ),
            Route(
                agent_name="strict-close",
                description="strict conservative SOX close",
                condition="strict",
            ),
        ],
        fallback_agent_name="lenient-close",
        router_config=RouterConfig(routing_strategy="hybrid"),
        model=model,
    )
    return router, [strict, lenient]


# --------------------------------------------------------------------------- #
# 9 · Debate  (advocate CLEAN vs CONTROL_ISSUE; judge computes deterministically)
# --------------------------------------------------------------------------- #
def build_debate(tools, te, model):
    pro = _agent(
        "close-debate-pro",
        "You argue the close is materially FINE under the firm's $5M threshold "
        "(small items are immaterial). 2-3 sentences.",
        tools,
        te,
        model,
    )
    con = _agent(
        "close-debate-con",
        "You argue the close has a CONTROL ISSUE: the $2M D1 misstatement affects the "
        "balance and must not be waived. 2-3 sentences.",
        tools,
        te,
        model,
    )
    judge = _agent(
        "close-debate-judge",
        "You are the judge — the deterministic books decide, not rhetoric.\n"
        + _GET_THRESHOLD
        + "Call compute_consolidation(threshold_usd=n) and end with EXACTLY:\n"
        + _DECIDE_TAIL,
        tools,
        te,
        model,
        {"compute_consolidation"},
    )
    debate = DebateAgent(name="close-debate", pro_agent=pro, con_agent=con, judge_agent=judge)
    return debate, [pro, con, judge]


# --------------------------------------------------------------------------- #
# 10 · Handoff  (triage → officer → controller; controller computes the verdict)
# --------------------------------------------------------------------------- #
def build_handoff(tools, te, model):
    controller = _agent(
        "ho-controller",
        "You are the controller and make the FINAL close call.\n"
        "1. Call get_materiality_policy for the threshold.\n"
        "2. Call compute_consolidation with that threshold_usd.\n"
        "3. Then decide. End with EXACTLY:\n" + _DECIDE_TAIL,
        tools,
        te,
        model,
        {"get_materiality_policy", "compute_consolidation"},
    )
    officer = BaseAgent(
        name="ho-close-officer",
        instructions="You are a close-intake officer. You do NOT decide — the controller does. "
        "One tool per turn:\n1. Call get_close_data.\n2. Then immediately call "
        "handoff_to_ho-controller. Never write a text answer.",
        model=model,
        tools=_tools(tools, {"get_close_data"}),
        tool_executor=te,
        handoffs=[
            Handoff(
                target_agent="ho-controller",
                description="Escalate for the final call.",
                return_to_parent=False,
            )
        ],
        memory_config=_MEM,
        config=_CFG,
    )
    triage = BaseAgent(
        name="ho-triage",
        instructions="Every request is a month-end close — immediately hand off to ho-close-officer.",
        model=model,
        handoffs=[
            Handoff(
                target_agent="ho-close-officer",
                description="Route to the officer.",
                return_to_parent=False,
            )
        ],
        memory_config=_MEM,
        config=_CFG,
    )
    return triage, [triage, officer, controller]


BUILDERS = {
    "sequential": build_sequential,
    "supervised": build_supervised,
    "planning": build_planning,
    "loop": build_loop,
    "reflection": build_reflection,
    "parallel": build_parallel,
    "scatter": build_scatter,
    "router": build_router,
    "debate": build_debate,
    "handoff": build_handoff,
}
