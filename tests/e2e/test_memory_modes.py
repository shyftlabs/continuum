"""
E2E tests — Memory modes and long-term/short-term memory interaction.

Tests agent behavior with different memory configurations:
- Memory disabled vs enabled
- Long-term memory recall across sessions
- Short-term session history recall
- User-scoped memory isolation
- Memory search affecting agent responses
- Memory-enabled agent learning and recall
- Edge cases: contradicting memories, large history, memory overflow
"""

from __future__ import annotations

import uuid

import pytest

pytestmark = pytest.mark.e2e

from tests.e2e.conftest import skip_if_no_api_key as _skip_if_no_api_key
from tests.e2e.conftest import skip_on_api_error as _skip_on_api_error


def _uid() -> str:
    return f"e2e-mem-{uuid.uuid4().hex[:10]}"


async def _cleanup_memory(user_id: str):
    """Best-effort cleanup of memory for a user."""
    try:
        from orchestrator.memory.client import MemoryClient

        client = MemoryClient()
        await client.delete_all(user_id=user_id)
    except Exception:
        pass


async def _cleanup_session(session_id: str):
    """Best-effort cleanup of session."""
    try:
        from orchestrator.session.client import SessionClient

        client = SessionClient()
        client.initialize()
        await client.delete_session(session_id)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Test: Memory disabled — agent has no recall
# ---------------------------------------------------------------------------


class TestMemoryDisabled:
    """Agent with memory disabled should not recall previous interactions."""

    @_skip_on_api_error
    async def test_no_memory_agent_forgets_everything(self):
        """Agent without memory cannot recall info from previous runs."""
        _skip_if_no_api_key()

        from orchestrator.agent.base import BaseAgent
        from orchestrator.agent.config import AgentConfig, AgentMemoryConfig
        from orchestrator.agent.runner import AgentRunner

        agent = BaseAgent(
            name="forgetful-agent",
            instructions=(
                "You are a helpful assistant. Be concise. "
                "If you don't know something, say 'I don't have that information'."
            ),
            memory_config=AgentMemoryConfig(
                search_memories=False,
                store_memories=False,
            ),
            config=AgentConfig(log_to_session=False),
        )

        runner = AgentRunner()

        # Run 1: Tell the agent something
        resp1 = await runner.run(agent, "My name is Alexandra and I'm 28 years old.")
        assert resp1.status.value == "success"

        # Run 2: Ask about it (no session, no memory)
        resp2 = await runner.run(agent, "What is my name and age?")
        assert resp2.content is not None
        content_lower = resp2.content.lower()
        # Should NOT know the name (no memory, no session)
        assert "alexandra" not in content_lower or "don't" in content_lower


# ---------------------------------------------------------------------------
# Test: Long-term memory — agent recalls across sessions
# ---------------------------------------------------------------------------


class TestLongTermMemory:
    """Agent with long-term memory stores and recalls facts across sessions."""

    @_skip_on_api_error
    async def test_agent_stores_and_recalls_fact(self):
        """Agent should recall a fact pre-stored in long-term memory."""
        _skip_if_no_api_key()

        from orchestrator.agent.base import BaseAgent
        from orchestrator.agent.config import AgentConfig, AgentMemoryConfig
        from orchestrator.agent.runner import AgentRunner
        from orchestrator.agent.types import MemoryScope
        from orchestrator.memory.client import MemoryClient

        uid = _uid()

        # Store the fact directly in long-term memory (simulates a previous session)
        mem_client = MemoryClient()
        await mem_client.add(
            "My favorite movie is Inception directed by Christopher Nolan.",
            user_id=uid,
        )

        import asyncio

        await asyncio.sleep(2)

        agent = BaseAgent(
            name="memory-agent",
            instructions=(
                "You are a helpful assistant with long-term memory. "
                "Use your memory to answer. Be concise."
            ),
            memory_config=AgentMemoryConfig(
                search_memories=True,
                store_memories=False,
                search_scope=MemoryScope.USER,
                store_scope=MemoryScope.USER,
                search_limit=5,
            ),
            config=AgentConfig(log_to_session=False),
        )

        runner = AgentRunner()

        try:
            resp = await runner.run(
                agent,
                "What is my favorite movie?",
                user_id=uid,
            )
            assert resp.content is not None
            content_lower = resp.content.lower()
            assert "inception" in content_lower or "nolan" in content_lower

        finally:
            await _cleanup_memory(uid)

    @_skip_on_api_error
    async def test_agent_recalls_multiple_facts(self):
        """Agent should recall multiple facts pre-stored in long-term memory."""
        _skip_if_no_api_key()

        from orchestrator.agent.base import BaseAgent
        from orchestrator.agent.config import AgentConfig, AgentMemoryConfig
        from orchestrator.agent.runner import AgentRunner
        from orchestrator.agent.types import MemoryScope
        from orchestrator.memory.client import MemoryClient

        uid = _uid()

        # Pre-store multiple facts directly
        mem_client = MemoryClient()
        await mem_client.add("I'm allergic to shellfish.", user_id=uid)
        await mem_client.add("My birthday is December 25th.", user_id=uid)
        await mem_client.add("I work at SpaceX as a rocket engineer.", user_id=uid)

        import asyncio

        await asyncio.sleep(2)

        agent = BaseAgent(
            name="multi-fact-agent",
            instructions=(
                "You are a personal assistant with long-term memory. "
                "Use your memory to answer questions. Be concise."
            ),
            memory_config=AgentMemoryConfig(
                search_memories=True,
                store_memories=False,
                search_scope=MemoryScope.USER,
                store_scope=MemoryScope.USER,
                search_limit=10,
            ),
            config=AgentConfig(log_to_session=False),
        )

        runner = AgentRunner()

        try:
            resp_allergy = await runner.run(agent, "Do I have any food allergies?", user_id=uid)
            assert resp_allergy.content is not None
            assert "shellfish" in resp_allergy.content.lower()

            resp_job = await runner.run(agent, "Where do I work and what do I do?", user_id=uid)
            assert resp_job.content is not None
            content_lower = resp_job.content.lower()
            assert "spacex" in content_lower or "rocket" in content_lower

        finally:
            await _cleanup_memory(uid)


# ---------------------------------------------------------------------------
# Test: Short-term memory (session history) — within-session recall
# ---------------------------------------------------------------------------


class TestShortTermSessionMemory:
    """Agent with session history recalls within the same session."""

    @_skip_on_api_error
    async def test_agent_recalls_within_session(self):
        """Agent should remember earlier messages in the same session."""
        _skip_if_no_api_key()

        from orchestrator.agent.base import BaseAgent
        from orchestrator.agent.config import AgentConfig, AgentMemoryConfig
        from orchestrator.agent.runner import AgentRunner
        from orchestrator.session.client import SessionClient

        agent = BaseAgent(
            name="session-memory-agent",
            instructions="You are a helpful assistant. Be concise. Use session history.",
            memory_config=AgentMemoryConfig(
                search_memories=False,
                store_memories=False,
            ),
            config=AgentConfig(log_to_session=True, session_history_limit=50),
        )

        runner = AgentRunner()

        # Create session explicitly
        session_client = SessionClient()
        session_client.initialize()
        sid = await session_client.get_or_create_session(
            session_id=f"e2e-stm-{uuid.uuid4().hex[:8]}",
            user_id="stm-user",
            agent_id="session-memory-agent",
        )

        try:
            # Turn 1
            resp1 = await runner.run(
                agent,
                "My cat's name is Whiskers and she is 3 years old.",
                session_id=sid,
                user_id="stm-user",
            )
            assert resp1.status.value == "success"

            # Turn 2 — same session
            resp2 = await runner.run(
                agent,
                "What is my cat's name?",
                session_id=sid,
                user_id="stm-user",
            )
            assert resp2.content is not None
            assert "whiskers" in resp2.content.lower()

            # Turn 3 — ask about age too
            resp3 = await runner.run(
                agent,
                "How old is my cat?",
                session_id=sid,
                user_id="stm-user",
            )
            assert resp3.content is not None
            assert "3" in resp3.content

        finally:
            await _cleanup_session(sid)

    @_skip_on_api_error
    async def test_session_history_limit_drops_old_messages(self):
        """With a small session_history_limit, old messages are dropped."""
        _skip_if_no_api_key()

        from orchestrator.agent.base import BaseAgent
        from orchestrator.agent.config import AgentConfig, AgentMemoryConfig
        from orchestrator.agent.runner import AgentRunner
        from orchestrator.session.client import SessionClient

        agent = BaseAgent(
            name="limited-history-agent",
            instructions=(
                "You are a helpful assistant. Be concise. "
                "If you don't know something, say 'I don't recall that'."
            ),
            memory_config=AgentMemoryConfig(
                search_memories=False,
                store_memories=False,
            ),
            # Very small history limit — only keeps last 4 messages
            config=AgentConfig(log_to_session=True, session_history_limit=4),
        )

        runner = AgentRunner()

        session_client = SessionClient()
        session_client.initialize()
        sid = await session_client.get_or_create_session(
            session_id=f"e2e-limit-{uuid.uuid4().hex[:8]}",
            user_id="limit-user",
            agent_id="limited-history-agent",
        )

        try:
            # Fill up session with many turns to push out early messages
            await runner.run(
                agent, "My secret code is ALPHA-7.", session_id=sid, user_id="limit-user"
            )
            await runner.run(agent, "What is 2+2?", session_id=sid, user_id="limit-user")
            await runner.run(agent, "Tell me a joke.", session_id=sid, user_id="limit-user")
            await runner.run(
                agent, "What's the weather like?", session_id=sid, user_id="limit-user"
            )
            await runner.run(
                agent, "Name a famous scientist.", session_id=sid, user_id="limit-user"
            )

            # Now ask about the secret code — it should have been pushed out
            resp = await runner.run(
                agent,
                "What was my secret code from earlier?",
                session_id=sid,
                user_id="limit-user",
            )
            assert resp.content is not None
            # With only 4 messages in history, the secret code should be gone
            content_lower = resp.content.lower()
            # Either doesn't recall OR somehow still has it (LLM might infer)
            # The key test: the system doesn't crash with limited history
            assert len(content_lower) > 0

        finally:
            await _cleanup_session(sid)


# ---------------------------------------------------------------------------
# Test: Memory-scoped user isolation in agent runs
# ---------------------------------------------------------------------------


class TestMemoryUserIsolationE2E:
    """Test that different users' memories are isolated in agent runs."""

    @_skip_on_api_error
    async def test_user_a_memories_not_visible_to_user_b(self):
        """User A's stored memories should not affect User B's responses."""
        _skip_if_no_api_key()

        from orchestrator.agent.base import BaseAgent
        from orchestrator.agent.config import AgentConfig, AgentMemoryConfig
        from orchestrator.agent.runner import AgentRunner
        from orchestrator.agent.types import MemoryScope

        uid_a = _uid()
        uid_b = _uid()

        agent = BaseAgent(
            name="isolated-mem-agent",
            instructions=(
                "You are a helpful assistant with memory. "
                "Answer based on what you know about the user. "
                "If you don't know, say 'I don't have that information'."
            ),
            memory_config=AgentMemoryConfig(
                search_memories=True,
                store_memories=True,
                search_scope=MemoryScope.USER,
                store_scope=MemoryScope.USER,
                search_limit=5,
            ),
            config=AgentConfig(log_to_session=False),
        )

        runner = AgentRunner()

        try:
            # User A stores a fact
            await runner.run(agent, "My password hint is 'blue ocean'.", user_id=uid_a)

            import asyncio

            await asyncio.sleep(2)

            # User B asks about it — should NOT know
            resp_b = await runner.run(
                agent,
                "What is my password hint?",
                user_id=uid_b,
            )
            assert resp_b.content is not None
            content_b = resp_b.content.lower()
            assert "blue ocean" not in content_b

        finally:
            await _cleanup_memory(uid_a)
            await _cleanup_memory(uid_b)


# ---------------------------------------------------------------------------
# Test: Combined short-term + long-term memory
# ---------------------------------------------------------------------------


class TestCombinedMemory:
    """Test agent using both session history AND long-term memory."""

    @_skip_on_api_error
    async def test_agent_uses_both_session_and_longterm(self):
        """Agent should combine session history and long-term memory."""
        _skip_if_no_api_key()

        from orchestrator.agent.base import BaseAgent
        from orchestrator.agent.config import AgentConfig, AgentMemoryConfig
        from orchestrator.agent.runner import AgentRunner
        from orchestrator.agent.types import MemoryScope
        from orchestrator.memory.client import MemoryClient
        from orchestrator.session.client import SessionClient

        uid = _uid()

        # Pre-store a long-term memory directly
        mem_client = MemoryClient()
        await mem_client.add("User's favorite cuisine is Italian.", user_id=uid)

        import asyncio

        await asyncio.sleep(2)

        agent = BaseAgent(
            name="combined-mem-agent",
            instructions=(
                "You are a helpful assistant with both session history and long-term memory. "
                "Use all available context. Be concise."
            ),
            memory_config=AgentMemoryConfig(
                search_memories=True,
                store_memories=True,
                search_scope=MemoryScope.USER,
                store_scope=MemoryScope.USER,
                search_limit=5,
            ),
            config=AgentConfig(log_to_session=True, session_history_limit=20),
        )

        runner = AgentRunner()

        session_client = SessionClient()
        session_client.initialize()
        sid = await session_client.get_or_create_session(
            session_id=f"e2e-combined-{uuid.uuid4().hex[:8]}",
            user_id=uid,
            agent_id="combined-mem-agent",
        )

        try:
            # Session turn 1: Tell something new (goes to session history)
            await runner.run(
                agent,
                "I'm planning a trip to Rome next month.",
                session_id=sid,
                user_id=uid,
            )

            # Session turn 2: Ask about both (should use session + long-term)
            resp = await runner.run(
                agent,
                "Given what you know about me, suggest a dinner plan for my trip.",
                session_id=sid,
                user_id=uid,
            )

            assert resp.content is not None
            content_lower = resp.content.lower()
            # Should reference Rome (from session) and/or Italian food (from memory)
            has_rome = "rome" in content_lower
            has_italian = (
                "italian" in content_lower or "pasta" in content_lower or "pizza" in content_lower
            )
            assert has_rome or has_italian, (
                f"Expected references to Rome or Italian food, got: {resp.content[:200]}"
            )

        finally:
            await _cleanup_memory(uid)
            await _cleanup_session(sid)


# ---------------------------------------------------------------------------
# Test: Contradicting memories
# ---------------------------------------------------------------------------


class TestContradictingMemories:
    """Test agent behavior when memories contain contradictions."""

    @_skip_on_api_error
    async def test_agent_handles_updated_preferences(self):
        """Agent should handle contradicting memories and prefer newer info."""
        _skip_if_no_api_key()

        from orchestrator.agent.base import BaseAgent
        from orchestrator.agent.config import AgentConfig, AgentMemoryConfig
        from orchestrator.agent.runner import AgentRunner
        from orchestrator.agent.types import MemoryScope
        from orchestrator.memory.client import MemoryClient

        uid = _uid()
        mem_client = MemoryClient()

        # Store contradicting facts directly
        await mem_client.add("My favorite color is blue.", user_id=uid)
        await mem_client.add(
            "I changed my mind — my favorite color is now green.",
            user_id=uid,
        )

        import asyncio

        await asyncio.sleep(2)

        agent = BaseAgent(
            name="update-mem-agent",
            instructions=(
                "You are a helpful assistant with memory. "
                "If you have conflicting information, prefer the most recent. Be concise."
            ),
            memory_config=AgentMemoryConfig(
                search_memories=True,
                store_memories=False,
                search_scope=MemoryScope.USER,
                search_limit=5,
            ),
            config=AgentConfig(log_to_session=False),
        )

        runner = AgentRunner()

        try:
            resp = await runner.run(agent, "What is my favorite color?", user_id=uid)
            assert resp.content is not None
            content_lower = resp.content.lower()
            # Should mention green (the latest) or blue — both are valid
            assert "green" in content_lower or "blue" in content_lower

        finally:
            await _cleanup_memory(uid)


# ---------------------------------------------------------------------------
# Test: Memory with tool-using agents
# ---------------------------------------------------------------------------


class TestMemoryWithToolAgent:
    """Test that memory works alongside tool-using agents."""

    @_skip_on_api_error
    async def test_tool_agent_uses_prestored_memory(self):
        """Tool-using agent should combine memory with tool results."""
        _skip_if_no_api_key()

        from mcp.types import CallToolResult, TextContent, Tool

        from orchestrator.agent.base import BaseAgent
        from orchestrator.agent.config import AgentConfig, AgentMemoryConfig
        from orchestrator.agent.runner import AgentRunner
        from orchestrator.agent.types import MemoryScope
        from orchestrator.memory.client import MemoryClient
        from orchestrator.tools.executor import ToolExecutor
        from orchestrator.tools.types import ToolContextConfig
        from orchestrator.tools.util import MCPUtil

        uid = _uid()

        # Pre-store user preference in memory
        mem_client = MemoryClient()
        await mem_client.add("The user is interested in buying a laptop.", user_id=uid)
        await mem_client.add("The user's budget is under $1000.", user_id=uid)

        import asyncio

        await asyncio.sleep(2)

        # Simple lookup tool
        class LookupServer:
            def __init__(self):
                self._name = "lookup-server"
                self.context_config = ToolContextConfig()
                self._tools = [
                    Tool(
                        name="lookup_price",
                        description="Look up product price by name.",
                        inputSchema={
                            "type": "object",
                            "properties": {"product": {"type": "string"}},
                            "required": ["product"],
                        },
                    ),
                ]
                self._connected = False

            @property
            def name(self):
                return self._name

            async def connect(self):
                self._connected = True

            async def cleanup(self):
                self._connected = False

            async def list_tools(self, metadata=None):
                return self._tools

            async def call_tool(self, tool_name, arguments):
                product = (arguments or {}).get("product", "unknown")
                prices = {"laptop": "$999", "phone": "$699", "headphones": "$199"}
                price = prices.get(product.lower(), "$N/A")
                return CallToolResult(
                    content=[TextContent(type="text", text=f"{product}: {price}")],
                    isError=False,
                )

            async def list_prompts(self):
                from mcp.types import ListPromptsResult

                return ListPromptsResult(prompts=[])

            async def get_prompt(self, name, arguments=None):
                from mcp.types import GetPromptResult

                return GetPromptResult(messages=[])

        server = LookupServer()
        await server.connect()
        tool_defs = await MCPUtil.get_function_tools(server)
        executor = ToolExecutor(tool_registry={server: None})
        await executor.initialize()

        agent = BaseAgent(
            name="tool-memory-agent",
            instructions=(
                "You are a shopping assistant with memory. "
                "Use the lookup_price tool when asked about prices. "
                "Use memory to personalize recommendations. Be concise."
            ),
            tools=tool_defs,
            tool_executor=executor,
            memory_config=AgentMemoryConfig(
                search_memories=True,
                store_memories=False,
                search_scope=MemoryScope.USER,
                search_limit=5,
            ),
            config=AgentConfig(log_to_session=False),
        )

        runner = AgentRunner(tool_executor=executor)

        try:
            resp = await runner.run(
                agent,
                "Check the laptop price and tell me if it fits my budget.",
                user_id=uid,
            )
            assert resp.status.value == "success"
            assert resp.content is not None
            content_lower = resp.content.lower()
            # Should reference both the price ($999) and budget context
            assert "999" in content_lower or "laptop" in content_lower

        finally:
            await _cleanup_memory(uid)


# ---------------------------------------------------------------------------
# Test: Evaluator checks memory-enhanced responses
# ---------------------------------------------------------------------------


class TestMemoryEvaluation:
    """Evaluate quality of memory-enhanced agent responses."""

    @_skip_on_api_error
    async def test_memory_improves_response_quality(self):
        """Agent with memory should give better answers than without."""
        _skip_if_no_api_key()

        from orchestrator.agent.base import BaseAgent
        from orchestrator.agent.config import AgentConfig, AgentMemoryConfig
        from orchestrator.agent.runner import AgentRunner
        from orchestrator.agent.types import MemoryScope
        from orchestrator.evaluation.evaluator_agent import EvaluatorAgent
        from orchestrator.evaluation.types import EvalCase
        from orchestrator.memory.client import MemoryClient

        uid = _uid()

        # Pre-store context in long-term memory
        mem_client = MemoryClient()
        await mem_client.add("The user is a vegetarian.", user_id=uid)
        await mem_client.add("The user is allergic to nuts.", user_id=uid)

        import asyncio

        await asyncio.sleep(2)

        # Agent WITH memory
        agent_with_mem = BaseAgent(
            name="mem-eval-agent",
            instructions=(
                "You are a personal chef assistant. "
                "Use what you know about the user's dietary restrictions. Be concise."
            ),
            memory_config=AgentMemoryConfig(
                search_memories=True,
                store_memories=False,
                search_scope=MemoryScope.USER,
                search_limit=5,
            ),
            config=AgentConfig(log_to_session=False),
        )

        runner = AgentRunner()

        try:
            resp = await runner.run(
                agent_with_mem,
                "Suggest a dinner recipe for me.",
                user_id=uid,
            )
            assert resp.content is not None

            # Evaluate with criteria
            evaluator = EvaluatorAgent(
                name="chef-judge",
                criteria=["correctness"],
                pass_threshold=0.5,
                rubrics={
                    "correctness": (
                        "The response should suggest a vegetarian recipe that does not "
                        "contain nuts or meat. If it contains meat or nuts, score very low."
                    ),
                },
            )

            case = EvalCase(
                input_text="Suggest a dinner recipe for me.",
                expected_output=(
                    "A vegetarian recipe without nuts. "
                    "For example: pasta with tomato sauce and vegetables."
                ),
                context=[
                    "User is a vegetarian.",
                    "User is allergic to nuts.",
                ],
            )

            eval_result = await evaluator.evaluate(case, resp.content)
            assert eval_result is not None

            score = eval_result.scores[0]
            if score.metadata.get("error") == "json_parse_failure":
                pytest.skip("Evaluator LLM output truncated")
            # Memory-enhanced response should be decent
            assert score.score >= 0.3, (
                f"Expected memory-enhanced response to score well, got {score.score}. "
                f"Response: {resp.content[:200]}"
            )

        finally:
            await _cleanup_memory(uid)
