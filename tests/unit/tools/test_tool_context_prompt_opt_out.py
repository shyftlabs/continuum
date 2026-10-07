"""ToolContextConfig.inject_into_system_prompt=False keeps a server's captured
variables out of the prompt.

The field was documented -- "If True, inject captured variables into system
prompt for LLM awareness" -- and defaulted True, but nothing read it. Setting
it False changed nothing: the values still went to the model provider on every
turn, beside "IMPORTANT: A session already exists. Do NOT call create_session
again." A server whose tools receive the session id by injection
(ToolContextVariable.inject_into) never needs the model to see it, and had no
way to say so.

The flag is read when the prompt is built, from the agent's current server
configs, not recorded on the variable when it is captured. A session restored
from storage therefore follows today's config: switching the flag off takes
effect on the next turn, not after the variable happens to be captured again.

Capture and injection into tool arguments are unaffected. The flag decides
only what the model is told.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from continuum.agent.utils.context_utils import create_run_context
from continuum.tools.executor import ToolExecutor
from continuum.tools.types import ToolContextConfig, ToolContextState

_CART = "tl:conv-4f1a"
_ORDER = "ord-7781"

_SHOP_TOOL = {
    "type": "function",
    "function": {
        "name": "shop__view_cart",
        "description": "View the cart",
        "parameters": {"type": "object", "properties": {"session_id": {"type": "string"}}},
    },
}


def _server(name: str, *, inject: bool = True, namespace: str | None = None) -> MagicMock:
    server = MagicMock()
    server.name = name
    server.context_config = ToolContextConfig(inject_into_system_prompt=inject, namespace=namespace)
    return server


def _state(**namespaces: dict[str, str]) -> ToolContextState:
    state = ToolContextState()
    for namespace, variables in namespaces.items():
        for name, value in variables.items():
            state.set(namespace, name, value)
    return state


class TestTheStateRendersWithoutExcludedNamespaces:
    def test_an_excluded_namespace_is_left_out(self):
        state = _state(shop={"session_id": _CART}, orders={"order_id": _ORDER})
        text = state.to_prompt_context(exclude_namespaces={"shop"})
        assert _CART not in text and "[shop]" not in text
        assert _ORDER in text and "[orders]" in text

    def test_nothing_left_renders_nothing(self):
        """Not a header with no values under it."""
        state = _state(shop={"session_id": _CART})
        assert state.to_prompt_context(exclude_namespaces={"shop"}) is None

    def test_no_exclusion_is_unchanged(self):
        state = _state(shop={"session_id": _CART})
        assert state.to_prompt_context() == state.to_prompt_context(exclude_namespaces=())


class TestTheExecutorKnowsWhichNamespacesOptedOut:
    def test_a_server_that_opted_out_is_named_by_its_namespace(self):
        executor = ToolExecutor(
            tool_registry={_server("shop", inject=False): None, _server("orders"): None}
        )
        assert executor.namespaces_kept_out_of_prompt() == {"shop"}

    def test_a_namespace_override_is_what_is_reported(self):
        """Variables are stored under config.namespace when it is set, so that
        is the name the exclusion has to match."""
        executor = ToolExecutor(
            tool_registry={_server("shop-eu", inject=False, namespace="shop"): None}
        )
        assert executor.namespaces_kept_out_of_prompt() == {"shop"}

    def test_a_shared_namespace_is_hidden_if_any_server_asks(self):
        """Two servers sharing a namespace share its variables. One asking to
        keep them out of the prompt is a statement about those values; the
        other's default True is not a request to show them."""
        executor = ToolExecutor(
            tool_registry={
                _server("shop-eu", inject=False, namespace="shop"): None,
                _server("shop-us", namespace="shop"): None,
            }
        )
        assert executor.namespaces_kept_out_of_prompt() == {"shop"}

    def test_the_default_hides_nothing(self):
        executor = ToolExecutor(tool_registry={_server("shop"): None})
        assert executor.namespaces_kept_out_of_prompt() == set()


def _agent():
    from continuum.agent.base import BaseAgent
    from continuum.agent.config import AgentConfig

    agent = BaseAgent(name="shopper", instructions="agent prompt", config=AgentConfig())
    agent.tools = [_SHOP_TOOL]
    return agent


async def _system_text(state: ToolContextState, executor: ToolExecutor | None) -> str:
    from continuum.agent.execution.message_builder import MessageBuilder

    with patch("continuum.observability.decorators.observe", lambda **kw: lambda f: f):
        messages, _ = await MessageBuilder().prepare_messages(
            _agent(),
            "what's in my cart?",
            create_run_context(),
            tool_context_state=state,
            tool_executor=executor,
        )
    return "\n".join(m["content"] for m in messages if m["role"] == "system")


class TestThePromptHonoursTheFlag:
    async def test_an_opted_out_server_values_do_not_reach_the_model(self):
        executor = ToolExecutor(tool_registry={_server("shop", inject=False): None})
        text = await _system_text(_state(shop={"session_id": _CART}), executor)
        assert _CART not in text
        assert "Current tool context" not in text

    async def test_nor_does_the_session_exists_instruction(self):
        """It refers to a session id the model was not shown."""
        executor = ToolExecutor(tool_registry={_server("shop", inject=False): None})
        text = await _system_text(_state(shop={"session_id": _CART}), executor)
        assert "A session already exists" not in text

    async def test_the_instruction_follows_only_the_shown_namespaces(self):
        executor = ToolExecutor(
            tool_registry={_server("shop", inject=False): None, _server("orders"): None}
        )
        state = _state(shop={"session_id": _CART}, orders={"order_id": _ORDER})
        text = await _system_text(state, executor)
        assert _ORDER in text and _CART not in text
        assert "A session already exists" not in text, "the only session id is hidden"

    async def test_the_default_still_injects(self):
        """The case the injection exists for must not regress."""
        executor = ToolExecutor(tool_registry={_server("shop"): None})
        text = await _system_text(_state(shop={"session_id": _CART}), executor)
        assert _CART in text
        assert "A session already exists" in text


class TestTheRunnerPassesTheExecutorItUses:
    async def test_the_agents_executor_reaches_prepare_messages(self, monkeypatch):
        from continuum.agent.runner import AgentRunner
        from continuum.agent.types import AgentResponse, ResponseStatus

        runner = AgentRunner()
        captured: dict = {}

        async def spy_prepare(*args, **kwargs):
            captured.update(kwargs)
            return ([], 0)

        async def fake_loop(agent, messages, context, run_state):
            return AgentResponse(content="ok", agent_name=agent.name, status=ResponseStatus.SUCCESS)

        async def fake_finalize(*a, **k):
            return None

        monkeypatch.setattr(runner._message_builder, "prepare_messages", spy_prepare)
        monkeypatch.setattr(runner._executor, "execute_loop", fake_loop)
        monkeypatch.setattr(runner._finalizer, "finalize", fake_finalize)

        agent = _agent()
        agent.tool_executor = ToolExecutor(tool_registry={_server("shop", inject=False): None})
        await runner.run(agent, "hello", context=create_run_context())

        assert captured["tool_executor"] is agent.tool_executor
