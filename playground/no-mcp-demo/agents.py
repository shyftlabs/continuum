"""
Reusable agent definitions — no tools, no MCP.

All agents rely on LLM reasoning only.
"""

from __future__ import annotations

from orchestrator import AgentConfig, AgentMemoryConfig, BaseAgent


def make_researcher(model: str) -> BaseAgent:
    return BaseAgent(
        name="researcher",
        instructions=(
            "You are a research specialist. "
            "Given a question or topic, provide a thorough, well-structured answer. "
            "Cite key facts, dates, and context. Aim for 3-5 paragraphs."
        ),
        model=model,
        memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
        config=AgentConfig(log_to_session=False, session_history_turns=0),
    )


def make_writer(model: str) -> BaseAgent:
    return BaseAgent(
        name="writer",
        instructions=(
            "You are a skilled writer. "
            "Take the research or context provided and write clear, engaging content. "
            "Match the format requested (essay, summary, bullet points, email, etc.)."
        ),
        model=model,
        memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
        config=AgentConfig(log_to_session=False, session_history_turns=0),
    )


def make_editor(model: str) -> BaseAgent:
    return BaseAgent(
        name="editor",
        instructions=(
            "You are a precise editor. "
            "Review the draft and improve clarity, flow, and correctness. "
            "Fix grammar issues and tighten the language. Return the improved version."
        ),
        model=model,
        memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
        config=AgentConfig(log_to_session=False, session_history_turns=0),
    )


def make_summarizer(model: str) -> BaseAgent:
    return BaseAgent(
        name="summarizer",
        instructions=(
            "You are a summarization expert. "
            "Read all prior pipeline steps and produce a concise, user-friendly summary. "
            "Keep it under 4 sentences."
        ),
        model=model,
        memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
        config=AgentConfig(log_to_session=False, session_history_turns=0),
    )


def make_analyst(model: str) -> BaseAgent:
    return BaseAgent(
        name="analyst",
        instructions=(
            "You are an analytical thinker. "
            "Given a subtopic or question, provide a focused, insightful analysis. "
            "Be specific and avoid generic statements. 2-3 paragraphs."
        ),
        model=model,
        memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
        config=AgentConfig(log_to_session=False, session_history_turns=0),
    )


def make_fact_checker(model: str) -> BaseAgent:
    return BaseAgent(
        name="fact-checker",
        instructions=(
            "You are a fact-checking specialist. "
            "Evaluate statements for accuracy. Note what is well-established, uncertain, or likely wrong. "
            "Be honest about the limits of your knowledge."
        ),
        model=model,
        memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
        config=AgentConfig(log_to_session=False, session_history_turns=0),
    )


def make_support_agent(model: str) -> BaseAgent:
    return BaseAgent(
        name="support-agent",
        instructions=(
            "You are a helpful general assistant. "
            "Answer questions, help the user clarify their request, and provide guidance. "
            "If the user wants research or writing done, tell them to be more specific."
        ),
        model=model,
        memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
        config=AgentConfig(log_to_session=True),
    )
