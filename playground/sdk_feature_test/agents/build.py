"""
Build all agents for the Research & Report pipeline.

Accepts config and optional tool_executor/tools, memory_client so pipeline
can build workflow-only (no MCP/memory) or full (with MCP, memory, structured output).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from orchestrator import (
    AgentConfig,
    AgentMemoryConfig,
    BaseAgent,
    ReflectionAgent,
    ReflectionConfig,
    create_loop_agent,
    create_parallel_agent,
    create_router_agent,
    create_sequential_agent,
)
from orchestrator.agent.types import (
    FailStrategy,
    Handoff,
    MergeStrategy,
    MemoryScope,
    TerminationConfig,
    TerminationType,
)

if TYPE_CHECKING:
    from orchestrator.memory import MemoryClient
    from orchestrator.tools import ToolExecutor

from playground.sdk_feature_test.config import SDKFeatureTestConfig
from playground.sdk_feature_test.schemas import ReportSummary


def _base_agent_config(
    cfg: SDKFeatureTestConfig,
    log_to_session: bool | None = None,
) -> AgentConfig:
    if log_to_session is None:
        log_to_session = getattr(cfg, "enable_session", True)
    return AgentConfig(
        max_turns=cfg.max_turns,
        log_to_session=log_to_session,
    )


def _memory_config(cfg: SDKFeatureTestConfig, enabled: bool) -> AgentMemoryConfig:
    if not enabled:
        return AgentMemoryConfig(search_memories=False, store_memories=False)
    return AgentMemoryConfig(
        search_memories=True,
        store_memories=True,
        search_scope=MemoryScope.USER,
        store_scope=MemoryScope.USER,
        search_limit=cfg.memory_search_limit,
    )


# ---- Route agents (for router) ----
def build_research_route_agent(cfg: SDKFeatureTestConfig) -> BaseAgent:
    """Agent for 'research' route: deep research on a topic."""
    return BaseAgent(
        name="research",
        instructions="You are a research specialist. Given a topic or question, provide a concise research summary with key points and sources. Be factual and clear.",
        model=cfg.default_model,
        temperature=cfg.agent_temperature,
        config=_base_agent_config(cfg),
        memory_config=_memory_config(cfg, cfg.enable_memory),
    )


def build_summarize_route_agent(cfg: SDKFeatureTestConfig) -> BaseAgent:
    """Agent for 'summarize' route: summarize content."""
    return BaseAgent(
        name="summarize",
        instructions="You are a summarization specialist. Given any text or content, produce a clear, concise summary. Preserve key facts and conclusions.",
        model=cfg.default_model,
        temperature=cfg.agent_temperature,
        config=_base_agent_config(cfg),
        memory_config=_memory_config(cfg, cfg.enable_memory),
    )


def build_qa_route_agent(cfg: SDKFeatureTestConfig) -> BaseAgent:
    """Agent for 'qa' route: Q&A on content."""
    return BaseAgent(
        name="qa",
        instructions="You are a Q&A specialist. Answer questions about the given content clearly and accurately. If the content does not contain the answer, say so.",
        model=cfg.default_model,
        temperature=cfg.agent_temperature,
        config=_base_agent_config(cfg),
        memory_config=_memory_config(cfg, cfg.enable_memory),
    )


def build_router_agent(
    cfg: SDKFeatureTestConfig,
) -> BaseAgent:
    """Router that routes to research, summarize, or qa."""
    return create_router_agent(
        name="main-router",
        routes=[
            ("research", "Deep research on a topic; gather and synthesize information"),
            ("summarize", "Summarize long content or multiple pieces of text"),
            ("qa", "Answer questions about provided content"),
        ],
        fallback="research",
        strategy="llm",
        model=cfg.default_model,
    )


# ---- Sequential pipeline: researcher -> writer -> editor ----
def build_researcher_agent(
    cfg: SDKFeatureTestConfig,
    tools: list[dict[str, Any]],
    tool_executor: ToolExecutor | None,
) -> BaseAgent:
    """Researcher agent; uses MCP tools when tool_executor and tools are provided."""
    instructions = "You are a researcher. Given a topic, gather key points and facts. Use the echo tool to record a short research note if you have a finding to capture (e.g. echo message='key finding: ...'). Output a concise research summary."
    return BaseAgent(
        name="researcher",
        instructions=instructions,
        model=cfg.default_model,
        temperature=cfg.agent_temperature,
        tools=tools if tool_executor else [],
        tool_executor=tool_executor,
        config=_base_agent_config(cfg),
        memory_config=_memory_config(cfg, cfg.enable_memory),
    )


def build_writer_agent(
    cfg: SDKFeatureTestConfig,
    use_structured_output: bool,
) -> BaseAgent:
    """Writer agent; optionally uses ReportSummary structured output."""
    instructions = "You are a report writer. Given research or content, produce a short report. Include a title, a few section summaries, key findings, and a confidence score (0-1) for the report quality."
    return BaseAgent(
        name="writer",
        instructions=instructions,
        model=cfg.default_model,
        temperature=cfg.agent_temperature,
        config=_base_agent_config(cfg),
        memory_config=_memory_config(cfg, cfg.enable_memory),
        enable_json_mode=use_structured_output,
        output_schema=ReportSummary if use_structured_output else None,
    )


def build_editor_agent(
    cfg: SDKFeatureTestConfig,
    with_handoff: bool,
    fact_checker_agent: BaseAgent | None,
) -> BaseAgent:
    """Editor agent; optionally hands off to fact_checker."""
    handoffs = []
    if with_handoff and fact_checker_agent:
        handoffs = [
            Handoff(
                target_agent=fact_checker_agent.name,
                description="Hand off to fact-checker when the report needs verification or fact-checking.",
            ),
        ]
    return BaseAgent(
        name="editor",
        instructions="You are an editor. Polish the report for clarity and consistency. If you notice claims that need fact-checking, hand off to the fact_checker. Otherwise output the final polished report.",
        model=cfg.default_model,
        temperature=cfg.agent_temperature,
        handoffs=handoffs,
        config=_base_agent_config(cfg),
        memory_config=_memory_config(cfg, cfg.enable_memory),
    )


def build_sequential_agent(
    cfg: SDKFeatureTestConfig,
    tools: list[dict[str, Any]],
    tool_executor: ToolExecutor | None,
    use_structured_output: bool,
    with_handoff: bool,
    fact_checker_agent: BaseAgent | None,
) -> BaseAgent:
    """Sequential pipeline: researcher -> writer -> editor."""
    writer = build_writer_agent(cfg, use_structured_output)
    editor = build_editor_agent(cfg, with_handoff, fact_checker_agent)
    researcher = build_researcher_agent(cfg, tools, tool_executor)
    return create_sequential_agent(
        name="research-pipeline",
        agents=[researcher, writer, editor],
        fail_strategy=FailStrategy.FAIL_FAST,
    )


# ---- Parallel: analyst_a, analyst_b ----
def build_analyst_agents(cfg: SDKFeatureTestConfig) -> list[BaseAgent]:
    """Two analyst agents for parallel execution."""
    return [
        BaseAgent(
            name="analyst_a",
            instructions="You are analyst A. Given a report or summary, analyze it from a critical perspective: strengths, gaps, and risks. Output a short analysis.",
            model=cfg.default_model,
            temperature=cfg.agent_temperature,
            config=_base_agent_config(cfg),
            memory_config=_memory_config(cfg, False),
        ),
        BaseAgent(
            name="analyst_b",
            instructions="You are analyst B. Given a report or summary, analyze it from a practical perspective: applicability, next steps, and recommendations. Output a short analysis.",
            model=cfg.default_model,
            temperature=cfg.agent_temperature,
            config=_base_agent_config(cfg),
            memory_config=_memory_config(cfg, False),
        ),
    ]


def build_parallel_agents(cfg: SDKFeatureTestConfig) -> BaseAgent:
    """Parallel agent running analyst_a and analyst_b."""
    analysts = build_analyst_agents(cfg)
    return create_parallel_agent(
        name="parallel-analysts",
        agents=analysts,
        merge_strategy=MergeStrategy.CONCATENATE,
        fail_strategy=FailStrategy.CONTINUE_ON_ERROR,
        timeout=cfg.parallel_timeout_per_agent,
    )


# ---- Loop: refiner ----
def build_refiner_agent(cfg: SDKFeatureTestConfig) -> BaseAgent:
    """Refiner agent run in a loop until done."""
    return BaseAgent(
        name="refiner",
        instructions="You improve the given text. Make it clearer and more concise. If the text is already good, say so and output it unchanged. Output only the improved (or unchanged) text, no meta-commentary.",
        model=cfg.default_model,
        temperature=cfg.agent_temperature,
        config=_base_agent_config(cfg),
        memory_config=_memory_config(cfg, False),
    )


def build_loop_agent(cfg: SDKFeatureTestConfig) -> BaseAgent:
    """Loop agent that runs refiner until termination."""
    refiner = build_refiner_agent(cfg)
    return create_loop_agent(
        name="refinement-loop",
        agent=refiner,
        termination_type=TerminationType.LLM_DECISION,
        max_iterations=cfg.loop_max_iterations,
        termination_prompt="Is the text already clear and final? Reply with exactly 'COMPLETE' if done, or 'CONTINUE' if more refinement is needed.",
    )


# ---- Reasoning patterns ----

def build_two_pass_agent(cfg: SDKFeatureTestConfig) -> BaseAgent:
    """Agent with two-pass reasoning: silent think-first LLM call before responding."""
    return BaseAgent(
        name="two-pass-reasoner",
        instructions="You are a thoughtful analyst. Answer the question clearly and thoroughly.",
        model=cfg.default_model,
        temperature=cfg.agent_temperature,
        config=AgentConfig(
            max_turns=cfg.max_turns,
            reasoning_mode=True,
            log_to_session=False,
        ),
        memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
    )


def build_react_agent(
    cfg: SDKFeatureTestConfig,
    tools: list[Any] | None = None,
    tool_executor: Any | None = None,
) -> BaseAgent:
    """Agent with ReAct mode: reasons via think() tool before calling real tools."""
    return BaseAgent(
        name="react-agent",
        instructions="You are a problem-solving agent. Use tools to get real data. Think before acting.",
        model=cfg.default_model,
        temperature=cfg.agent_temperature,
        tools=tools or [],
        tool_executor=tool_executor,
        config=AgentConfig(
            max_turns=cfg.max_turns,
            react_mode=True,
            log_to_session=False,
        ),
        memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
    )


def build_normal_agent(
    cfg: SDKFeatureTestConfig,
    tools: list[Any] | None = None,
    tool_executor: Any | None = None,
) -> BaseAgent:
    """Same as react agent but without react_mode — for direct comparison."""
    return BaseAgent(
        name="normal-agent",
        instructions="You are a problem-solving agent. Use tools to get real data.",
        model=cfg.default_model,
        temperature=cfg.agent_temperature,
        tools=tools or [],
        tool_executor=tool_executor,
        config=AgentConfig(
            max_turns=cfg.max_turns,
            react_mode=False,
            log_to_session=False,
        ),
        memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
    )


def build_reflection_agent(cfg: SDKFeatureTestConfig) -> ReflectionAgent:
    """ReflectionAgent wrapping a writer agent: self-critiques and retries if needed."""
    inner = BaseAgent(
        name="reflection-inner-writer",
        instructions="You are a concise writer. Answer the question in 2-3 clear sentences.",
        model=cfg.default_model,
        temperature=cfg.agent_temperature,
        config=AgentConfig(max_turns=cfg.max_turns, log_to_session=False),
        memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
    )
    return ReflectionAgent(
        name="reflection-writer",
        agent=inner,
        reflection_config=ReflectionConfig(
            critique_prompt=(
                "Review the response above. Reply ONLY 'PASS' if it fully and clearly answers "
                "the request in 2-3 sentences, or 'NEEDS IMPROVEMENT: <reason>' if not."
            ),
            max_reflections=2,
        ),
        memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
    )


# ---- Handoff target ----
def build_fact_checker_agent(cfg: SDKFeatureTestConfig) -> BaseAgent:
    """Fact-checker agent (handoff target from editor)."""
    return BaseAgent(
        name="fact_checker",
        instructions="You are a fact-checker. Review the report for factual accuracy. Note any claims that need sources or corrections. Output a short fact-check note and suggest corrections if any.",
        model=cfg.default_model,
        temperature=0.3,
        config=_base_agent_config(cfg),
        memory_config=_memory_config(cfg, False),
    )


def build_question_detector_agent(cfg: SDKFeatureTestConfig) -> BaseAgent:
    """Condition agent: returns 'true' if input is a question, 'false' otherwise."""
    return BaseAgent(
        name="question_detector",
        instructions=(
            "Determine whether the user's input is a question. "
            "Reply with exactly one word: 'true' if it is a question, 'false' if it is not. "
            "No punctuation, no explanation, just 'true' or 'false'."
        ),
        model=cfg.default_model,
        temperature=0.0,
        config=_base_agent_config(cfg),
        memory_config=_memory_config(cfg, False),
    )
