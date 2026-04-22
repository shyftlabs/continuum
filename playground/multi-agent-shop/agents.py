"""
Reusable sub-agent definitions for workflow-shop.

All agents are constructed here and imported by workflows.py.
Each agent has a focused, single responsibility.
"""

from __future__ import annotations

from typing import Any

from orchestrator import AgentConfig, AgentMemoryConfig, AgentMemoryScope, BaseAgent


def make_search_agent(tools: list[dict[str, Any]], tool_executor: Any, model: str) -> BaseAgent:
    return BaseAgent(
        name="search-agent",
        instructions=(
            "You are a pet shop search specialist. "
            "Use search_products and get_product tools to find products matching the user's request. "
            "Always return product IDs, names, and prices clearly."
        ),
        model=model,
        tools=tools,
        tool_executor=tool_executor,
        memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
        config=AgentConfig(log_to_session=False),
    )


def make_recommend_agent(model: str) -> BaseAgent:
    return BaseAgent(
        name="recommend-agent",
        instructions=(
            "You are a pet product recommendation specialist. "
            "Given a list of search results, recommend the single best option with a clear reason. "
            "Always include the product ID in your recommendation."
        ),
        model=model,
        memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
        config=AgentConfig(log_to_session=False),
    )


def make_cart_agent(tools: list[dict[str, Any]], tool_executor: Any, model: str) -> BaseAgent:
    return BaseAgent(
        name="cart-agent",
        instructions=(
            "You are a pet shop cart specialist. "
            "Use add_to_cart, view_cart, and checkout tools to manage the user's cart. "
            "Always use the session_id provided in context for cart operations."
        ),
        model=model,
        tools=tools,
        tool_executor=tool_executor,
        memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
        config=AgentConfig(log_to_session=False),
    )


def make_summary_agent(model: str) -> BaseAgent:
    return BaseAgent(
        name="summary-agent",
        instructions=(
            "You are a friendly pet shop assistant. "
            "Read the prior pipeline steps from context and write a single, clear summary "
            "for the user: what was found, what was recommended, and what was done. "
            "Keep it less than 3-4 sentences."
        ),
        model=model,
        memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
        config=AgentConfig(log_to_session=False),
    )



def make_analyst_agent(model: str) -> BaseAgent:
    return BaseAgent(
        name="analyst-agent",
        instructions=(
            "You are a product value analyst. "
            "Given a product's details, assess its value for money, quality, and suitability. "
            "Be concise — 3-4 sentences max per product."
        ),
        model=model,
        memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
        config=AgentConfig(log_to_session=True, session_history_turns=0),
    )


def make_writer_agent(model: str) -> BaseAgent:
    return BaseAgent(
        name="writer-agent",
        instructions=(
            "You are a pet product copywriter. "
            "Write clear, friendly, and helpful content about pet products. "
            "Tailor your tone to the format requested (guide, email, summary, etc.)."
        ),
        model=model,
        memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
        config=AgentConfig(log_to_session=True),
    )


def make_support_agent(model: str) -> BaseAgent:
    return BaseAgent(
        name="support-agent",
        instructions=(
            "You are a pet care support agent. "
            "Answer general questions about pet care, nutrition, and product usage. "
            "If the user needs to search or buy something, tell them to ask the shop assistant."
        ),
        model=model,
        memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
        config=AgentConfig(log_to_session=True),
    )
