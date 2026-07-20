"""
Reusable sub-agent definitions for gateway-multi-agent-shop.

All agents accept gateway_mode so the Smart Gateway can route each
agent independently using its own tier (strict / modest / quality).
"""

from __future__ import annotations

from typing import Any

from continuum import AgentConfig, AgentMemoryConfig, BaseAgent


def make_search_agent(
    tools: list[dict[str, Any]], tool_executor: Any, model: str, gateway_mode: str | None = None
) -> BaseAgent:
    return BaseAgent(
        name="search-agent",
        instructions=(
            "You are a pet shop search specialist. "
            "Use search_products and get_product tools to find products matching the user's request. "
            "For a FULL inventory audit or stock report (e.g. 'audit the whole inventory', "
            "'full stock report'), call fetch_inventory — it returns every SKU with stock levels. "
            "To investigate recent order activity, anomalies, or errors (e.g. 'investigate recent "
            "orders', 'check the order logs'), call fetch_order_logs. "
            "For a service-config edit (e.g. 'raise the DB pool size'), call read with "
            "path='service.yaml', then write to save the change. If you are then asked for the "
            "ORIGINAL values that changed, retrieve them from the earlier config you read. "
            "Always return product IDs, names, and prices clearly."
        ),
        model=model,
        gateway_mode=gateway_mode,
        tools=tools,
        tool_executor=tool_executor,
        memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
        config=AgentConfig(log_to_session=False),
    )


def make_recommend_agent(model: str, gateway_mode: str | None = None) -> BaseAgent:
    return BaseAgent(
        name="recommend-agent",
        instructions=(
            "You are a pet product recommendation specialist. "
            "Given a list of search results, recommend the single best option with a clear reason. "
            "Always include the product ID in your recommendation."
        ),
        model=model,
        gateway_mode=gateway_mode,
        memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
        config=AgentConfig(log_to_session=False),
    )


def make_cart_agent(
    tools: list[dict[str, Any]], tool_executor: Any, model: str, gateway_mode: str | None = None
) -> BaseAgent:
    return BaseAgent(
        name="cart-agent",
        instructions=(
            "You are a pet shop cart specialist. "
            "Use add_to_cart, view_cart, and checkout tools to manage the user's cart."
        ),
        model=model,
        gateway_mode=gateway_mode,
        tools=tools,
        tool_executor=tool_executor,
        memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
        config=AgentConfig(log_to_session=False),
    )


def make_summary_agent(model: str, gateway_mode: str | None = None) -> BaseAgent:
    return BaseAgent(
        name="summary-agent",
        instructions=(
            "You are a friendly pet shop assistant. "
            "Read the prior pipeline steps from context and write a single, clear summary "
            "for the user: what was found, what was recommended, and what was done. "
            "Keep it less than 3-4 sentences."
        ),
        model=model,
        gateway_mode=gateway_mode,
        memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
        config=AgentConfig(log_to_session=False),
    )


def make_analyst_agent(
    model: str,
    gateway_mode: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_executor: Any = None,
) -> BaseAgent:
    instructions = (
        "You are a product value analyst. "
        "Given a product's details, assess its value for money, quality, and suitability. "
        "Be concise — 3-4 sentences max per product."
    )
    if tools:
        instructions += (
            " If your assigned task is to audit recent order activity or errors, first call "
            "fetch_order_logs to pull the raw order-service log, then analyze it. If asked to "
            "audit stock, call fetch_inventory. Base your analysis on the returned data."
        )
    return BaseAgent(
        name="analyst-agent",
        instructions=instructions,
        model=model,
        gateway_mode=gateway_mode,
        tools=tools or [],
        tool_executor=tool_executor,
        memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
        config=AgentConfig(log_to_session=True, session_history_turns=0),
    )


def make_writer_agent(model: str, gateway_mode: str | None = None) -> BaseAgent:
    return BaseAgent(
        name="writer-agent",
        instructions=(
            "You are a pet product copywriter. "
            "Write clear, friendly, and helpful content about pet products. "
            "Tailor your tone to the format requested (guide, email, summary, etc.)."
        ),
        model=model,
        gateway_mode=gateway_mode,
        memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
        config=AgentConfig(log_to_session=True),
    )


def make_support_agent(model: str, gateway_mode: str | None = None) -> BaseAgent:
    return BaseAgent(
        name="support-agent",
        instructions=(
            "You are a pet care support agent. "
            "Answer general questions about pet care, nutrition, and product usage. "
            "If the user needs to search or buy something, tell them to ask the shop assistant."
        ),
        model=model,
        gateway_mode=gateway_mode,
        memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
        config=AgentConfig(log_to_session=True),
    )
