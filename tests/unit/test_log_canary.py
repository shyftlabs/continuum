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

    _STANDARD = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)

    class Collector(logging.Handler):
        """Sees what a *third-party* handler sees, which is more than a formatter.

        Continuum's own formatters render ``getMessage()`` and ignore unknown
        record attributes, so ``logger.info(msg, extra={...})`` is invisible to
        them. Datadog's handler, python-json-logger and most structlog bridges
        serialise ``record.__dict__`` and would emit it. Capturing both channels
        keeps the canary honest about where content can actually escape.
        """

        def emit(self, record: logging.LogRecord) -> None:
            try:
                rendered.append(record.getMessage())
            except Exception as e:  # a malformed format string is its own bug
                rendered.append(f"<unrenderable record: {e}>")
            for key, value in record.__dict__.items():
                if key not in _STANDARD:
                    rendered.append(f"extra[{key}]={value!r}")

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


# ── workflows ─────────────────────────────────────────────────────────────────


def _response(content: str):
    from continuum.agent.types import AgentResponse, ResponseStatus

    return AgentResponse(content=content, status=ResponseStatus.SUCCESS, agent_name="branch")


def _llm_returning(content: str):
    """An LLM client whose chat() answers, so the merge path runs to the end."""
    reply = MagicMock()
    reply.content = content
    client = MagicMock()
    client.chat = AsyncMock(return_value=reply)
    return client


@pytest.mark.asyncio
class TestWorkflowMerge:
    """A merge prompt carries every branch's full output -- the most concentrated
    content in the codebase, and it is logged at INFO."""

    async def test_parallel_merge_prompt(self, logged):
        from continuum.agent.workflow.parallel import MergeStrategy, ParallelAgent, ParallelConfig

        agent = ParallelAgent(
            name="fan",
            agents=[BaseAgent(name="a", instructions="x")],
            parallel_config=ParallelConfig(merge_strategy=MergeStrategy.LLM_SUMMARIZE),
        )
        await agent._merge_results(
            {"a": _response(CANARY)}, input_text="summarise", llm_client=_llm_returning("done")
        )
        assert_clean(logged, "parallel._merge_results")

    async def test_parallel_merge_carries_the_original_input(self, logged):
        from continuum.agent.workflow.parallel import MergeStrategy, ParallelAgent, ParallelConfig

        agent = ParallelAgent(
            name="fan",
            agents=[BaseAgent(name="a", instructions="x")],
            parallel_config=ParallelConfig(merge_strategy=MergeStrategy.LLM_SUMMARIZE),
        )
        await agent._merge_results(
            {"a": _response("ok")}, input_text=CANARY, llm_client=_llm_returning("done")
        )
        assert_clean(logged, "parallel._merge_results(input)")

    async def test_scatter_merge_prompt(self, logged):
        from continuum.agent.workflow.scatter import ScatterAgent

        agent = ScatterAgent(name="scat", agents=[BaseAgent(name="a", instructions="x")])
        await agent._merge_results(
            {"a": _response(CANARY)},
            original_input="summarise",
            llm_client=_llm_returning("done"),
        )
        assert_clean(logged, "scatter._merge_results")


@pytest.mark.asyncio
class TestReflection:
    """The critique prompt is the model's own answer, handed back for review."""

    async def test_the_critiqued_response(self, logged):
        from continuum.agent.workflow.reflection import ReflectionAgent

        agent = ReflectionAgent(name="ref", agent=BaseAgent(name="a", instructions="x"))
        await agent._critique(CANARY, llm_client=_llm_returning('{"verdict": "ok"}'))
        assert_clean(logged, "reflection._critique")


# ── tool-attention routing ────────────────────────────────────────────────────


class TestToolAttention:
    """The router logs the query it routed on -- that is the user's question."""

    def test_the_routed_query(self, logged):
        from continuum.agent.types import RunContext
        from continuum.tools.tool_attention.router import ToolAttentionConfig, ToolAttentionRouter

        router = ToolAttentionRouter(ToolAttentionConfig(k=1, min_tools=1))
        # Stand in for the embedding registry: this test is about the log line,
        # not about which tools semantic search would pick.
        router._initialized = True
        router._registry = MagicMock(ready=True, search=MagicMock(return_value=["lookup"]))

        tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
        router.route([{"role": "user", "content": CANARY}], tools, RunContext(run_id="r"))
        assert_clean(logged, "tool_attention.route(query)")


# ── input scanners ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestInputScanner:
    """``reason`` comes from a callable the integrator supplies. The contract at
    AgentConfig does not say what it may contain, so a scanner that quotes the
    offending input satisfies it -- and this line runs at WARNING on the security
    path, which is exactly what gets forwarded and kept."""

    async def test_a_scanner_reason_that_quotes_the_input(self, logged):
        from continuum.agent.config import AgentConfig
        from continuum.agent.execution.message_builder import MessageBuilder
        from continuum.exceptions import InputBlockedError

        def nosy_scanner(text: str):
            return text, False, f"blocked: {text}"

        agent = BaseAgent(
            name="clinic", instructions="hi", config=AgentConfig(input_scanners=[nosy_scanner])
        )
        with pytest.raises(InputBlockedError):
            await MessageBuilder().prepare_messages(
                agent=agent, input=CANARY, context=RunContext(run_id="run-7")
            )
        assert_clean(logged, "message_builder input scanner reason")


# ── identity, on the channel no call site can reach ───────────────────────────


class TestIdentityInTheLogContext:
    """llm/callbacks.py publishes user_id and session_id into the logging
    context, and JSONFormatter stamps it onto every structured line -- so a
    single leaking value appears on thousands of lines, not one."""

    def test_the_context_does_not_carry_it_to_the_line(self, logged, monkeypatch):
        from continuum.config import settings
        from continuum.logging import LogContext, _context_for_output

        monkeypatch.setattr(settings, "session_id_secret", "0123456789abcdef" * 4)
        with LogContext(trace_id="t-1", user_id=CANARY, session_id=f"u:{CANARY}"):
            rendered = _context_for_output()
        assert CANARY not in str(rendered), rendered

    def test_a_real_run_does_not_stamp_the_caller(self, logged, monkeypatch):
        """set_log_context is what callbacks.py calls; this is that shape."""
        from continuum.config import settings
        from continuum.logging import _context_for_output, clear_log_context, set_log_context

        monkeypatch.setattr(settings, "session_id_secret", "0123456789abcdef" * 4)
        try:
            set_log_context(trace_id="t-1", user_id=CANARY, session_id=f"u:{CANARY}")
            assert CANARY not in str(_context_for_output())
        finally:
            clear_log_context()


# ── sessions ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestSessionPaths:
    """A session id is not an opaque handle. With SESSION_HASH_IDS=false -- the
    shipped default -- it is derived in plaintext from the user id, so every line
    that logs one writes the user id too, and the sentinel arrives at both.
    """

    async def _client(self):
        from continuum.session import SessionClient, SessionConfig
        from continuum.session.providers.memory import MemorySessionProvider

        cfg = SessionConfig(enabled=True, provider="memory", hash_session_ids=False)
        client = SessionClient(session_config=cfg, memory_client=None, auto_initialize=False)
        client.set_provider(MemorySessionProvider(cfg))
        client._initialized = True
        return client

    async def test_creating_a_session_does_not_log_the_user(self, logged, monkeypatch):
        from continuum.config import settings

        monkeypatch.setattr(settings, "session_id_secret", "0123456789abcdef" * 4)
        client = await self._client()
        await client.get_or_create_session(user_id=CANARY, conversation_id="conv-1")
        assert_clean(logged, "session.get_or_create_session")

    async def test_adding_a_message_does_not_log_the_session_id(self, logged, monkeypatch):
        """bind_principal because the session is owned -- PR #98's ownership gate
        refuses an unbound caller, and the refusal would end the test before the
        write ever logged anything."""
        from continuum.config import settings
        from continuum.llm.types import ChatMessage
        from continuum.session import bind_principal

        monkeypatch.setattr(settings, "session_id_secret", "0123456789abcdef" * 4)
        client = await self._client()
        with bind_principal(CANARY):
            sid = await client.get_or_create_session(user_id=CANARY)
            logged.clear()
            await client.add_message(
                sid, ChatMessage(role="user", content="hi"), store_in_memory=False
            )
        assert_clean(logged, "session.add_message")

    async def test_reading_history_does_not_log_the_session_id(self, logged, monkeypatch):
        from continuum.config import settings
        from continuum.session import bind_principal

        monkeypatch.setattr(settings, "session_id_secret", "0123456789abcdef" * 4)
        client = await self._client()
        with bind_principal(CANARY):
            sid = await client.get_or_create_session(user_id=CANARY)
            logged.clear()
            await client.get_conversation_history(sid)
        assert_clean(logged, "session.get_conversation_history")


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
