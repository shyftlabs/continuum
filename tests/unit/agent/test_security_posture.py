"""
Step 1 — agent security-posture check.

When an agent has side-effectful ("sensitive") tools but no authorization is
configured (no policy_store and no config.access_policies), construction should
surface it: a loud warning by default, or a hard error under strict_security.
Benign-only agents, and agents that DO wire authorization, stay silent.

The warning goes through continuum's logger, whose ``propagate`` is False, so we
spy on ``base._logger.warning`` rather than relying on pytest's ``caplog``.
"""

from __future__ import annotations

import pytest

import continuum.agent.base as base_mod
from continuum.agent.base import BaseAgent
from continuum.agent.config import AgentConfig
from continuum.agent.exceptions import AgentConfigurationError
from continuum.security.policy import AccessPolicy, PolicyStore


@pytest.fixture
def warnings(monkeypatch):
    """Capture messages passed to base._logger.warning."""
    captured: list[str] = []
    monkeypatch.setattr(
        base_mod._logger, "warning", lambda msg, *a, **k: captured.append(str(msg))
    )
    return captured


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "description": f"{name} tool", "parameters": {}},
    }


class TestSecurityPostureWarning:
    def test_sensitive_tool_without_auth_warns(self, warnings):
        BaseAgent(
            name="support",
            instructions="help",
            tools=[_tool("delete_account"), _tool("get_status")],
        )
        assert any(
            "delete_account" in m and "UNAUTHORIZED" in m for m in warnings
        ), warnings

    def test_benign_tools_do_not_warn(self, warnings):
        BaseAgent(
            name="weatherbot",
            instructions="help",
            tools=[_tool("get_weather"), _tool("lookup_city")],
        )
        assert warnings == []

    def test_policy_store_suppresses_warning(self, warnings):
        BaseAgent(
            name="support",
            instructions="help",
            tools=[_tool("delete_account")],
            policy_store=PolicyStore.default_deny(),
        )
        assert warnings == []

    def test_access_policies_suppress_warning(self, warnings):
        cfg = AgentConfig(
            access_policies=[
                AccessPolicy(
                    name="p", subjects=["*"], resources=["tool:*"], effect="allow"
                )
            ]
        )
        BaseAgent(
            name="support",
            instructions="help",
            tools=[_tool("delete_account")],
            config=cfg,
        )
        assert warnings == []


class TestSecurityPostureStrict:
    def test_strict_security_raises_on_sensitive_tool_without_auth(self):
        with pytest.raises(AgentConfigurationError, match="UNAUTHORIZED"):
            BaseAgent(
                name="support",
                instructions="help",
                tools=[_tool("send_payment")],
                config=AgentConfig(strict_security=True),
            )

    def test_strict_security_ok_when_authorized(self):
        # No raise when a policy store is present.
        agent = BaseAgent(
            name="support",
            instructions="help",
            tools=[_tool("send_payment")],
            policy_store=PolicyStore.default_deny(),
            config=AgentConfig(strict_security=True),
        )
        assert agent.name == "support"

    def test_strict_security_ok_when_only_benign_tools(self):
        agent = BaseAgent(
            name="weatherbot",
            instructions="help",
            tools=[_tool("get_weather")],
            config=AgentConfig(strict_security=True),
        )
        assert agent.name == "weatherbot"
