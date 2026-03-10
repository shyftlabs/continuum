"""
Unit tests for prompt engineering features on BaseAgent:

- PromptTemplate  : {slot} resolution from template_vars, context, and built-ins
- Few-shot examples : examples[] injected into the system prompt
- Instruction modifiers : callables that mutate the prompt at runtime
- MessageBuilder integration : resolve_system_prompt(context) called at execution time
- Edge cases : unknown slots, empty fields, modifier errors, clone/to_dict
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.agent.base import BaseAgent
from orchestrator.agent.types import RunContext, generate_run_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_ctx(**kwargs) -> RunContext:
    return RunContext(run_id=generate_run_id(), **kwargs)


def make_agent(**kwargs) -> BaseAgent:
    return BaseAgent(name="test-agent", **kwargs)


# ===========================================================================
# PromptTemplate — {slot} resolution
# ===========================================================================


class TestPromptTemplate:
    def test_template_var_resolved(self):
        agent = make_agent(
            instructions="You are helping {user_name}.",
            template_vars={"user_name": "Alice"},
        )
        result = agent.resolve_system_prompt(None)
        assert result == "You are helping Alice."

    def test_builtin_date_resolved(self):
        from datetime import date

        agent = make_agent(instructions="Today is {date}.")
        result = agent.resolve_system_prompt(None)
        assert result == f"Today is {date.today().isoformat()}."

    def test_builtin_agent_name_resolved(self):
        agent = make_agent(instructions="I am {agent_name}.")
        result = agent.resolve_system_prompt(None)
        assert result == "I am test-agent."

    def test_context_user_id_resolved(self):
        agent = make_agent(instructions="User: {user_id}.")
        ctx = make_ctx(user_id="u-99")
        result = agent.resolve_system_prompt(ctx)
        assert result == "User: u-99."

    def test_context_session_id_resolved(self):
        agent = make_agent(instructions="Session: {session_id}.")
        ctx = make_ctx(session_id="sess-42")
        result = agent.resolve_system_prompt(ctx)
        assert result == "Session: sess-42."

    def test_context_run_id_resolved(self):
        agent = make_agent(instructions="Run: {run_id}.")
        ctx = make_ctx()
        result = agent.resolve_system_prompt(ctx)
        assert ctx.run_id in result

    def test_context_metadata_resolved(self):
        agent = make_agent(instructions="Plan: {plan_type}.")
        ctx = make_ctx(metadata={"plan_type": "enterprise"})
        result = agent.resolve_system_prompt(ctx)
        assert result == "Plan: enterprise."

    def test_template_vars_override_context_metadata(self):
        """template_vars have higher priority than context.metadata."""
        agent = make_agent(
            instructions="Tier: {tier}.",
            template_vars={"tier": "gold"},
        )
        ctx = make_ctx(metadata={"tier": "bronze"})
        result = agent.resolve_system_prompt(ctx)
        assert result == "Tier: gold."

    def test_template_vars_override_builtin_slots(self):
        """template_vars can override even built-in slots like {date}."""
        agent = make_agent(
            instructions="Date: {date}.",
            template_vars={"date": "2099-01-01"},
        )
        result = agent.resolve_system_prompt(None)
        assert result == "Date: 2099-01-01."

    def test_unknown_slot_preserved(self):
        agent = make_agent(instructions="Hello {unknown_slot}.")
        result = agent.resolve_system_prompt(None)
        assert "{unknown_slot}" in result

    def test_no_slots_unchanged(self):
        agent = make_agent(instructions="Plain instructions.")
        result = agent.resolve_system_prompt(None)
        assert result == "Plain instructions."

    def test_none_context_uses_empty_strings_for_context_slots(self):
        agent = make_agent(instructions="U={user_id} S={session_id}.")
        result = agent.resolve_system_prompt(None)
        assert result == "U= S=."

    def test_multiple_slots_resolved_in_one_call(self):
        agent = make_agent(
            instructions="{greeting}, {user_name}! You are on {plan}.",
            template_vars={"greeting": "Hello", "user_name": "Bob", "plan": "pro"},
        )
        result = agent.resolve_system_prompt(None)
        assert result == "Hello, Bob! You are on pro."


# ===========================================================================
# Few-shot examples
# ===========================================================================


class TestFewShotExamples:
    def test_examples_appended_to_prompt(self):
        agent = make_agent(
            instructions="Classify sentiment.",
            examples=[
                {"input": "The sky is blue.", "output": "positive"},
                {"input": "The food was terrible.", "output": "negative"},
            ],
        )
        result = agent.resolve_system_prompt(None)
        assert "Examples:" in result
        assert "The sky is blue." in result
        assert "positive" in result
        assert "The food was terrible." in result
        assert "negative" in result

    def test_examples_come_after_instructions(self):
        agent = make_agent(
            instructions="Base instructions.",
            examples=[{"input": "Q", "output": "A"}],
        )
        result = agent.resolve_system_prompt(None)
        assert result.index("Base instructions.") < result.index("Examples:")

    def test_empty_examples_no_injection(self):
        agent = make_agent(instructions="No examples here.", examples=[])
        result = agent.resolve_system_prompt(None)
        assert "Examples:" not in result
        assert result == "No examples here."

    def test_single_example(self):
        agent = make_agent(
            instructions="Translate.",
            examples=[{"input": "Hello", "output": "Hola"}],
        )
        result = agent.resolve_system_prompt(None)
        assert "Input: Hello" in result
        assert "Output: Hola" in result

    def test_examples_combined_with_template_vars(self):
        agent = make_agent(
            instructions="Help {user_name}.",
            template_vars={"user_name": "Carol"},
            examples=[{"input": "Hi", "output": "Hello"}],
        )
        result = agent.resolve_system_prompt(None)
        assert "Help Carol." in result
        assert "Examples:" in result

    def test_multiple_examples_all_present(self):
        exs = [{"input": f"q{i}", "output": f"a{i}"} for i in range(5)]
        agent = make_agent(instructions="Answer.", examples=exs)
        result = agent.resolve_system_prompt(None)
        for i in range(5):
            assert f"q{i}" in result
            assert f"a{i}" in result


# ===========================================================================
# Instruction modifiers
# ===========================================================================


class TestInstructionModifiers:
    def test_single_modifier_applied(self):
        def add_footer(prompt: str, ctx) -> str:
            return prompt + " [footer]"

        agent = make_agent(
            instructions="Base.",
            instruction_modifiers=[add_footer],
        )
        result = agent.resolve_system_prompt(None)
        assert result == "Base. [footer]"

    def test_modifier_receives_context(self):
        def add_tier(prompt: str, ctx) -> str:
            tier = ctx.metadata.get("user_tier", "free") if ctx else "free"
            return prompt + f" tier={tier}"

        agent = make_agent(
            instructions="Support.",
            instruction_modifiers=[add_tier],
        )
        ctx = make_ctx(metadata={"user_tier": "enterprise"})
        result = agent.resolve_system_prompt(ctx)
        assert "tier=enterprise" in result

    def test_modifier_receives_none_context_safely(self):
        def safe_modifier(prompt: str, ctx) -> str:
            return prompt + (" [no-ctx]" if ctx is None else " [ctx]")

        agent = make_agent(instructions="Base.", instruction_modifiers=[safe_modifier])
        assert "[no-ctx]" in agent.resolve_system_prompt(None)
        assert "[ctx]" in agent.resolve_system_prompt(make_ctx())

    def test_modifiers_applied_in_order(self):
        results = []

        def m1(prompt: str, ctx) -> str:
            results.append("m1")
            return prompt + " m1"

        def m2(prompt: str, ctx) -> str:
            results.append("m2")
            return prompt + " m2"

        agent = make_agent(instructions="Start.", instruction_modifiers=[m1, m2])
        result = agent.resolve_system_prompt(None)
        assert result == "Start. m1 m2"
        assert results == ["m1", "m2"]

    def test_modifier_error_does_not_raise(self):
        def bad_modifier(prompt: str, ctx) -> str:
            raise RuntimeError("modifier failure")

        agent = make_agent(instructions="Safe.", instruction_modifiers=[bad_modifier])
        # Should not raise; returns the prompt as it was before the failed modifier
        result = agent.resolve_system_prompt(None)
        assert result == "Safe."

    def test_modifier_after_failed_modifier_still_runs(self):
        def bad(prompt: str, ctx) -> str:
            raise ValueError("oops")

        def good(prompt: str, ctx) -> str:
            return prompt + " good"

        agent = make_agent(instructions="Start.", instruction_modifiers=[bad, good])
        result = agent.resolve_system_prompt(None)
        assert result == "Start. good"

    def test_no_modifiers_returns_base_prompt(self):
        agent = make_agent(instructions="No modifiers.", instruction_modifiers=[])
        assert agent.resolve_system_prompt(None) == "No modifiers."

    def test_modifier_uses_session_length(self):
        """Modifier can adapt prompt based on session history length."""

        def verbose_if_new(prompt: str, ctx) -> str:
            if ctx and not ctx.session_id:
                return prompt + " Welcome! Let me introduce myself."
            return prompt

        agent = make_agent(instructions="Hi.", instruction_modifiers=[verbose_if_new])

        new_user_ctx = make_ctx(session_id=None)
        existing_ctx = make_ctx(session_id="existing-session")

        assert "Welcome!" in agent.resolve_system_prompt(new_user_ctx)
        assert "Welcome!" not in agent.resolve_system_prompt(existing_ctx)


# ===========================================================================
# Layering — all three features together
# ===========================================================================


class TestAllThreeLayers:
    def test_template_then_examples_then_modifier(self):
        """All three layers apply in correct order."""

        def append_tag(prompt: str, ctx) -> str:
            return prompt + " [tagged]"

        agent = make_agent(
            instructions="Help {user_name}.",
            template_vars={"user_name": "Dave"},
            examples=[{"input": "Hello", "output": "Hi"}],
            instruction_modifiers=[append_tag],
        )
        result = agent.resolve_system_prompt(None)

        assert "Help Dave." in result        # template resolved
        assert "Examples:" in result         # examples injected
        assert "[tagged]" in result          # modifier applied
        # Order: instructions → examples → modifier tag at end
        assert result.index("Help Dave.") < result.index("Examples:")
        assert result.endswith("[tagged]")

    def test_modifier_sees_post_template_prompt(self):
        """The modifier receives the already-template-resolved prompt."""
        received = []

        def capture(prompt: str, ctx) -> str:
            received.append(prompt)
            return prompt

        agent = make_agent(
            instructions="Hello {name}.",
            template_vars={"name": "Eve"},
            instruction_modifiers=[capture],
        )
        agent.resolve_system_prompt(None)
        assert "Hello Eve." in received[0]
        assert "{name}" not in received[0]

    def test_modifier_sees_post_examples_prompt(self):
        """The modifier receives the prompt with examples already appended."""
        received = []

        def capture(prompt: str, ctx) -> str:
            received.append(prompt)
            return prompt

        agent = make_agent(
            instructions="Base.",
            examples=[{"input": "Q", "output": "A"}],
            instruction_modifiers=[capture],
        )
        agent.resolve_system_prompt(None)
        assert "Examples:" in received[0]


# ===========================================================================
# system_prompt property — backward compatibility
# ===========================================================================


class TestSystemPromptProperty:
    def test_plain_instructions_unchanged(self):
        agent = make_agent(instructions="Simple.")
        assert agent.system_prompt == "Simple."

    def test_template_vars_resolved_without_context(self):
        agent = make_agent(
            instructions="Hi {name}.",
            template_vars={"name": "Frank"},
        )
        assert agent.system_prompt == "Hi Frank."

    def test_context_slots_empty_without_context(self):
        agent = make_agent(instructions="User={user_id}.")
        assert agent.system_prompt == "User=."

    def test_examples_present_in_property(self):
        agent = make_agent(
            instructions="Base.",
            examples=[{"input": "Q", "output": "A"}],
        )
        assert "Examples:" in agent.system_prompt

    def test_modifier_applied_in_property(self):
        agent = make_agent(
            instructions="Base.",
            instruction_modifiers=[lambda p, _: p + " end"],
        )
        assert agent.system_prompt == "Base. end"


# ===========================================================================
# MessageBuilder integration
# ===========================================================================


class TestMessageBuilderIntegration:
    @pytest.mark.asyncio
    async def test_template_vars_resolved_in_messages(self):
        from orchestrator.agent.execution.message_builder import MessageBuilder

        agent = make_agent(
            instructions="Serving {user_id}.",
            template_vars={},
        )
        ctx = make_ctx(user_id="u-777")
        ctx.session_id = None

        builder = MessageBuilder()
        messages = await builder.prepare_messages(agent=agent, input="hi", context=ctx)

        system_msgs = [m["content"] for m in messages if m["role"] == "system"]
        assert any("u-777" in c for c in system_msgs)

    @pytest.mark.asyncio
    async def test_examples_injected_in_messages(self):
        from orchestrator.agent.execution.message_builder import MessageBuilder

        agent = make_agent(
            instructions="Classify.",
            examples=[{"input": "Good", "output": "positive"}],
        )
        ctx = make_ctx()
        ctx.session_id = None

        builder = MessageBuilder()
        messages = await builder.prepare_messages(agent=agent, input="test", context=ctx)

        system_msgs = [m["content"] for m in messages if m["role"] == "system"]
        assert any("Examples:" in c for c in system_msgs)
        assert any("positive" in c for c in system_msgs)

    @pytest.mark.asyncio
    async def test_modifier_applied_in_messages(self):
        from orchestrator.agent.execution.message_builder import MessageBuilder

        def tier_tag(prompt: str, ctx) -> str:
            tier = ctx.metadata.get("tier", "free") if ctx else "free"
            return prompt + f" [tier:{tier}]"

        agent = make_agent(
            instructions="Help.",
            instruction_modifiers=[tier_tag],
        )
        ctx = make_ctx(metadata={"tier": "premium"})
        ctx.session_id = None

        builder = MessageBuilder()
        messages = await builder.prepare_messages(agent=agent, input="hello", context=ctx)

        system_msgs = [m["content"] for m in messages if m["role"] == "system"]
        assert any("[tier:premium]" in c for c in system_msgs)


# ===========================================================================
# clone() and to_dict()
# ===========================================================================


class TestCloneAndToDict:
    def test_clone_preserves_template_vars(self):
        agent = make_agent(instructions="Hi.", template_vars={"k": "v"})
        cloned = agent.clone(name="cloned-agent")
        assert cloned.template_vars == {"k": "v"}

    def test_clone_allows_override_of_template_vars(self):
        agent = make_agent(instructions="Hi.", template_vars={"k": "v"})
        cloned = agent.clone(name="cloned-agent", template_vars={"k": "new"})
        assert cloned.template_vars == {"k": "new"}

    def test_clone_preserves_examples(self):
        exs = [{"input": "Q", "output": "A"}]
        agent = make_agent(instructions="Hi.", examples=exs)
        cloned = agent.clone(name="cloned-agent")
        assert cloned.examples == exs

    def test_clone_preserves_instruction_modifiers(self):
        m = lambda p, c: p
        agent = make_agent(instructions="Hi.", instruction_modifiers=[m])
        cloned = agent.clone(name="cloned-agent")
        assert cloned.instruction_modifiers == [m]

    def test_to_dict_includes_template_vars(self):
        agent = make_agent(instructions="Hi.", template_vars={"x": 1})
        d = agent.to_dict()
        assert d["template_vars"] == {"x": 1}

    def test_to_dict_includes_examples(self):
        exs = [{"input": "Q", "output": "A"}]
        agent = make_agent(instructions="Hi.", examples=exs)
        d = agent.to_dict()
        assert d["examples"] == exs

    def test_to_dict_has_instruction_modifiers_flag(self):
        agent_no_mod = make_agent(instructions="Hi.")
        agent_with_mod = make_agent(
            instructions="Hi.", instruction_modifiers=[lambda p, c: p]
        )
        assert agent_no_mod.to_dict()["has_instruction_modifiers"] is False
        assert agent_with_mod.to_dict()["has_instruction_modifiers"] is True

    def test_clone_is_independent_copy(self):
        """Mutating clone's template_vars doesn't affect original."""
        agent = make_agent(instructions="Hi.", template_vars={"k": "v"})
        cloned = agent.clone(name="cloned-agent")
        cloned.template_vars["k"] = "changed"
        assert agent.template_vars["k"] == "v"
