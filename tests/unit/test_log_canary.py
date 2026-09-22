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

import contextlib
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


# ── long-term memory ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestMemoryPaths:
    """A search query is what the user asked. A stored memory is what they said
    last week. Both were logged with a slice -- query[:100], query[:50] -- which
    leaks a prefix and discards the rest."""

    async def _client(self, monkeypatch):
        from continuum.memory.base import BaseMemoryProvider
        from continuum.memory.client import MemoryClient
        from continuum.memory.config import MemoryConfig

        provider = MagicMock(spec=BaseMemoryProvider)
        provider.search = AsyncMock(return_value=MagicMock(results=[], total_results=0))
        provider.add = AsyncMock(return_value=MagicMock(results=[], message="ok"))
        client = MemoryClient(config=MemoryConfig(), provider=provider, auto_initialize=False)
        client._initialized = True
        return client

    async def test_a_search_query_is_not_logged(self, logged, monkeypatch):
        client = await self._client(monkeypatch)
        await client.search(CANARY, user_id="u1")
        assert_clean(logged, "memory.client.search(query)")

    async def test_the_searching_user_is_not_logged(self, logged, monkeypatch):
        from continuum.config import settings

        monkeypatch.setattr(settings, "session_id_secret", "0123456789abcdef" * 4)
        client = await self._client(monkeypatch)
        await client.search("what did I say?", user_id=CANARY)
        assert_clean(logged, "memory.client.search(user_id)")


# ── the llm client's session handling ─────────────────────────────────────────


@pytest.mark.asyncio
class TestLLMClientSessionId:
    """The client logs which session it loaded history from, on every call that
    has one -- the highest-frequency site in the package. With
    SESSION_HASH_IDS=false that id contains the user id.

    Driven through the real chat() rather than by calling the logger directly:
    a test that wraps the value itself would pass whatever client.py does.
    """

    async def test_loading_history_does_not_log_the_session_id(self, logged, monkeypatch):
        from continuum.config import settings
        from continuum.llm.client import LLMClient
        from continuum.llm.config import LLMConfig

        monkeypatch.setattr(settings, "session_id_secret", "0123456789abcdef" * 4)

        session_client = MagicMock()
        session_client.is_enabled = True
        session_client.get_conversation_history = AsyncMock(
            return_value=[{"role": "user", "content": "earlier"}]
        )
        container = MagicMock(session_client=session_client)
        # Imported inside the function, so patch it where it is defined.
        monkeypatch.setattr("continuum.core.container.get_container", lambda: container)

        client = LLMClient(config=LLMConfig(model="nonexistent-model-xyz"), enable_langfuse=False)
        with contextlib.suppress(Exception):
            # The provider call after the history load is expected to fail; the
            # line under test has already been emitted by then.
            await client.chat(
                messages=[{"role": "user", "content": "hi"}],
                session_id=f"u:{CANARY}",
                auto_session=True,
            )

        assert_clean(logged, "llm.client history load")


# ── tools ─────────────────────────────────────────────────────────────────────


class TestToolContextCapture:
    """A tool result that will not parse is logged with a preview. The preview is
    the tool's output -- a record, a balance, a search hit."""

    def test_an_unparseable_result_is_not_previewed(self, logged):
        from continuum.tools.executor import ToolExecutor
        from continuum.tools.types import ToolContextConfig, ToolContextVariable

        server = MagicMock()
        server.name = "srv"
        server.context_config = ToolContextConfig(
            variables=[ToolContextVariable(name="x", capture_from="lookup", json_path="$.x")]
        )

        ToolExecutor()._capture_context_variables(server, "lookup", f"not json: {CANARY}")
        assert_clean(logged, "tools.executor._capture_context_variables")


@pytest.mark.asyncio
class TestMCPToolInvocation:
    """Malformed arguments are logged whole on the way in. They are whatever the
    model asked for on the user's behalf."""

    async def test_malformed_arguments_are_not_logged(self, logged):
        from continuum.tools.util import MCPUtil

        server = MagicMock()
        server.name = "srv"
        tool = MagicMock()
        tool.name = "lookup"

        with contextlib.suppress(Exception):
            await MCPUtil.invoke_mcp_tool_with_artifact(
                server=server,
                tool=tool,
                input_json=f'{{"q": "{CANARY}"',  # unterminated
            )
        assert_clean(logged, "tools.util.invoke_mcp_tool_with_artifact")


# ── the agent's memory service ────────────────────────────────────────────────


@pytest.mark.asyncio
class TestAgentMemoryService:
    """Two values, both the user's: the query the run searches memory with, and
    the memories that come back. Both were logged with a slice."""

    async def _service(self, returned):
        from continuum.agent.services.memory_service import MemoryService

        memory_client = MagicMock()
        memory_client.is_enabled = True
        memory_client.search = AsyncMock(
            return_value=MagicMock(results=returned, total_results=len(returned))
        )
        return MemoryService(memory_client=memory_client)

    async def test_the_search_query_is_not_logged(self, logged, monkeypatch):
        from continuum.config import settings

        monkeypatch.setattr(settings, "session_id_secret", "0123456789abcdef" * 4)
        agent = BaseAgent(name="clinic", instructions="hi")
        agent.memory_config.search_memories = True
        service = await self._service([])
        await service.retrieve_memories(agent, CANARY, RunContext(run_id="r", user_id="u1"))
        assert_clean(logged, "memory_service.retrieve_memories(query)")

    async def test_the_owning_user_of_a_memory_is_not_logged(self, logged, monkeypatch):
        """Each recalled row is logged with the user it belongs to, to prove
        isolation -- which means proving it by printing the person."""
        from continuum.config import settings

        monkeypatch.setattr(settings, "session_id_secret", "0123456789abcdef" * 4)
        agent = BaseAgent(name="clinic", instructions="hi")
        agent.memory_config.search_memories = True
        service = await self._service(
            [MagicMock(memory="a note", user_id=CANARY, metadata={}, score=0.9)]
        )
        await service.retrieve_memories(agent, "recall", RunContext(run_id="r", user_id="u1"))
        assert_clean(logged, "memory_service.retrieve_memories(memory owner)")

    async def test_a_returned_memory_is_not_logged(self, logged, monkeypatch):
        """The memory text itself, coming back out of long-term storage."""
        from continuum.config import settings

        monkeypatch.setattr(settings, "session_id_secret", "0123456789abcdef" * 4)
        agent = BaseAgent(name="clinic", instructions="hi")
        agent.memory_config.search_memories = True
        # score must be a real float or None: the site formats it with :.3f, and
        # a MagicMock there diverts the code before the memory text is logged.
        service = await self._service(
            [MagicMock(memory=CANARY, user_id="u1", metadata={}, score=0.9)]
        )
        await service.retrieve_memories(agent, "recall", RunContext(run_id="r", user_id="u1"))
        assert_clean(logged, "memory_service.retrieve_memories(results)")


# ── credentials ───────────────────────────────────────────────────────────────


class TestCredentialsNeverReachTheLog:
    """A different canary, for a different kind of secret.

    mem0's config embeds live provider API keys -- the embedder's and the
    fact-extraction LLM's -- and the provider logged the whole dict at DEBUG.
    Not user content, so log_content() is the wrong tool: an operator wants to
    see which provider, which model, which host. Only the credentials must go,
    which is what redact_dict already does everywhere else.
    """

    FAKE_KEY = "sk-svcacct-ZZQX-FAKE-KEY-NEVER-LOG-ME"

    def test_the_mem0_config_is_masked_before_logging(self, logged):
        from continuum.memory.providers.mem0 import _loggable_config

        config = {
            "version": "v1.1",
            "llm": {"provider": "gemini", "config": {"model": "x", "api_key": self.FAKE_KEY}},
            "embedder": {"provider": "openai", "config": {"api_key": self.FAKE_KEY}},
        }
        rendered = str(_loggable_config(config))
        assert self.FAKE_KEY not in rendered

    def test_the_useful_parts_survive(self, logged):
        """Masking must not cost the diagnostic: which provider and model is the
        reason this line exists."""
        from continuum.memory.providers.mem0 import _loggable_config

        rendered = str(
            _loggable_config(
                {
                    "llm": {
                        "provider": "gemini",
                        "config": {"model": "gemini-2.5-flash", "api_key": self.FAKE_KEY},
                    }
                }
            )
        )
        assert "gemini" in rendered
        assert "gemini-2.5-flash" in rendered

    def test_it_is_not_governed_by_LOG_PROMPT_CONTENT(self, monkeypatch):
        """A credential is not a debugging convenience. Turning content logging
        on must not turn keys back on."""
        from continuum.config import settings
        from continuum.memory.providers.mem0 import _loggable_config

        monkeypatch.setattr(settings, "log_prompt_content", True)
        config = {"embedder": {"config": {"api_key": self.FAKE_KEY}}}
        assert self.FAKE_KEY not in str(_loggable_config(config))


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
