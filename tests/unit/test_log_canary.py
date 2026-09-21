"""No path may put the user's own words into a log line.

``PromptContentFilter`` can only withhold what a call site declares with
``log_content()``. Declaring is a human act, and humans forget -- silently, and
forever, because a leaked log line looks exactly like a working one.

This suite makes forgetting loud. Each test drives a real path with one
sentinel string and asserts the sentinel is absent from everything logged. It
does not care *how* a leak happens: a site nobody converted, an f-string written
next year, a helper that stringifies a payload on its way past. The assertion is
the same either way, and the test name says which path failed.

Why the records are read *before* the filter runs, deliberately:

    A collector attached here sees what the call site actually handed to the
    logger, not what survived the handlers. That is stricter than an operator's
    view, and it is the view we want. A site that leaks 46 characters of PHI is
    a bug even when a length heuristic happens to catch it at 96 -- the heuristic
    cannot be relied on, and relying on it is how a short leak ships unnoticed.

So a failure here means: wrap that value in ``log_content()``.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from continuum.agent import BaseAgent
from continuum.agent.types import RunContext
from continuum.config import settings

# Distinctive enough that a substring match cannot be a coincidence, and
# greppable in a log file when one of these does fail.
CANARY = "ZZQX-CANARY-NEVER-LOG-ME-7f3a"


@pytest.fixture
def logged(monkeypatch):
    """Everything logged anywhere under ``continuum``, rendered as text.

    Attached to the root ``continuum`` logger, so records from every child
    module arrive here by propagation.
    """
    monkeypatch.setattr(settings, "log_prompt_content", False)

    rendered: list[str] = []

    class Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                rendered.append(record.getMessage())
            except Exception as e:  # a malformed format string is its own bug
                rendered.append(f"<unrenderable record: {e}>")

    root = logging.getLogger("continuum")
    handler = Collector()
    saved_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)  # DEBUG too: a leak at DEBUG is still a leak
    try:
        yield rendered
    finally:
        root.removeHandler(handler)
        root.setLevel(saved_level)


def assert_clean(rendered: list[str], path: str) -> None:
    assert rendered, f"{path}: nothing was logged -- this test is not exercising the path"
    offenders = [line for line in rendered if CANARY in line]
    assert not offenders, (
        f"{path} logged the user's input.\n"
        f"Wrap the value in log_content() at the site that produced:\n  "
        + "\n  ".join(line[:200] for line in offenders)
    )


# ── prompt assembly ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestPromptAssembly:
    """Everything that lands in the prompt: instructions, memories, history,
    RAG context, and the input itself."""

    async def test_the_user_input(self, logged):
        from continuum.agent.execution.message_builder import MessageBuilder

        agent = BaseAgent(name="clinic", instructions="You are a triage bot.")
        await MessageBuilder().prepare_messages(
            agent=agent, input=CANARY, context=RunContext(run_id="run-1")
        )
        assert_clean(logged, "message_builder.prepare_messages(input)")

    async def test_a_retrieved_memory(self, logged):
        """A stored note is the user's words too, arriving a week later."""
        from continuum.agent.execution.message_builder import MessageBuilder
        from continuum.agent.services.memory_service import MemoryService

        memory = MagicMock(spec=MemoryService)
        memory.search_memories = AsyncMock(return_value=[{"memory": CANARY}])

        agent = BaseAgent(name="clinic", instructions="hi")
        agent.memory_config.search_memories = True
        await MessageBuilder(memory_service=memory).prepare_messages(
            agent=agent, input="hello", context=RunContext(run_id="run-2", user_id="u1")
        )
        assert_clean(logged, "message_builder.prepare_messages(memories)")

    async def test_the_system_instructions(self, logged):
        """An operator's own prompt can hold a customer name or a policy."""
        from continuum.agent.execution.message_builder import MessageBuilder

        agent = BaseAgent(name="clinic", instructions=f"Treat {CANARY} with care.")
        await MessageBuilder().prepare_messages(
            agent=agent, input="hello", context=RunContext(run_id="run-3")
        )
        assert_clean(logged, "message_builder.prepare_messages(instructions)")


# ── tools ─────────────────────────────────────────────────────────────────────


def _tool_agent(returns: str) -> BaseAgent:
    """An agent whose executor returns ``returns``.

    A stub rather than a real ToolExecutor, which is built from MCP servers:
    what is under test is ToolService's logging, not tool dispatch, and the
    logging sites are the same either way.
    """

    class _Executor:
        async def execute_tool_calls(self, tool_calls, **kwargs):
            return [{"role": "tool", "content": returns, "tool_call_id": "call-1"}]

    return BaseAgent(name="clinic", instructions="hi", tool_executor=_Executor())


def _call(call_id: str, arguments: str) -> dict[str, Any]:
    """A tool call in the dict shape execute_tool_call accepts."""
    return {"id": call_id, "function": {"name": "lookup", "arguments": arguments}}


@pytest.mark.asyncio
class TestToolCalls:
    """Arguments go in carrying what the user asked for; results come back
    carrying whatever the tool read."""

    async def test_tool_arguments(self, logged):
        from continuum.agent.services.tool_service import ToolService

        agent = _tool_agent("ok")
        await ToolService().execute_tool_call(
            agent=agent,
            tool_call=_call("call-1", f'{{"query": "{CANARY}"}}'),
            context=RunContext(run_id="run-4"),
        )
        assert_clean(logged, "tool_service.execute_tool_call(arguments)")

    async def test_tool_results(self, logged):
        from continuum.agent.services.tool_service import ToolService

        agent = _tool_agent(f"record: {CANARY}")
        await ToolService().execute_tool_call(
            agent=agent,
            tool_call=_call("call-2", '{"query": "x"}'),
            context=RunContext(run_id="run-5"),
        )
        assert_clean(logged, "tool_service.execute_tool_call(result)")

    async def test_malformed_tool_arguments(self, logged):
        """The error path logs the raw argument string, which is the one most
        likely to be forgotten because it only runs when something is wrong."""
        from continuum.agent.services.tool_service import ToolService

        agent = _tool_agent("ok")
        await ToolService().execute_tool_call(
            agent=agent,
            tool_call=_call("call-3", f'{{"query": "{CANARY}"'),  # unterminated JSON
            context=RunContext(run_id="run-6"),
        )
        assert_clean(logged, "tool_service.execute_tool_call(malformed arguments)")


# ── context compression ───────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestHeadroomSidecar:
    """The whole conversation is handed to the compressor, and both the request
    and the response are logged at DEBUG."""

    async def _client(self, monkeypatch, response: dict[str, Any]):
        from continuum.llm.headroom.client import HeadroomClient

        client = HeadroomClient(api_base="http://sidecar.test")
        reply = MagicMock()
        reply.json = MagicMock(return_value=response)
        reply.raise_for_status = MagicMock()
        monkeypatch.setattr(client._client, "post", AsyncMock(return_value=reply))
        return client

    async def test_the_request(self, logged, monkeypatch):
        client = await self._client(
            monkeypatch, {"messages": [], "tokens_before": 10, "tokens_after": 5}
        )
        await client.compress([{"role": "user", "content": CANARY}], model="gpt-4o-mini")
        assert_clean(logged, "headroom.client.compress(request)")

    async def test_the_response(self, logged, monkeypatch):
        client = await self._client(
            monkeypatch,
            {
                "messages": [{"role": "user", "content": CANARY}],
                "tokens_before": 10,
                "tokens_after": 5,
            },
        )
        await client.compress([{"role": "user", "content": "x"}], model="gpt-4o-mini")
        assert_clean(logged, "headroom.client.compress(response)")


# ── the canary itself has to work ─────────────────────────────────────────────


class TestTheCanaryCatchesALeak:
    """A canary that cannot fail proves nothing. This is the control."""

    def test_an_unwrapped_value_is_caught(self, logged):
        logging.getLogger("continuum.probe").info("leaked: %s", CANARY)
        with pytest.raises(AssertionError, match="logged the user's input"):
            assert_clean(logged, "control")

    def test_a_wrapped_value_is_not(self, logged):
        from continuum.logging import log_content

        logging.getLogger("continuum.probe").info("withheld: %s", log_content(CANARY))
        assert_clean(logged, "control")
