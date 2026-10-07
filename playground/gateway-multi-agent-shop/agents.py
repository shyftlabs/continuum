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


def make_clarify_agent(model: str, gateway_mode: str | None = None) -> BaseAgent:
    """Where an unsure router sends a request: ask, don't guess.

    Used by router-system-one as its fallback, so a request the router could not
    place with confidence (or one about something else entirely) gets a question
    back instead of a confident answer from the wrong specialist. No tools: it
    asks, it does not act.
    """
    return BaseAgent(
        name="clarify-agent",
        instructions=(
            "You are the pet shop's front desk. The shop's router could not tell with "
            "confidence what this request needs, so do not try to answer or fulfil it. "
            "Ask one short clarifying question instead. First say in a few words what you "
            "think the user wants. Then say what the shop can do: search for products, "
            "manage the cart (add, view, checkout), or give pet-care advice. If the request "
            "needs more than one of these, for example finding a toy and then adding it to "
            "the cart, suggest doing them one at a time, search first. Each message is "
            "routed on its own, so ask the user to reply with one concrete request and "
            "suggest the exact wording, such as 'find a chew toy for my puppy'. If the "
            "request has nothing to do with pets or the shop, say so politely and offer "
            "what the shop can do. Keep it to two or three sentences."
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
