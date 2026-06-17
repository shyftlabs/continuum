"""
Verifies that RunContext.data_labels are wired into the tool-access policy check.

Before the fix, data_labels existed but no policy check ever read them. Now the
tool executor folds the run's active labels into the policy `subjects`, so a
project can write `deny(subjects=["pii"], resources=["tool:send_email"])` and
have it actually block a PII-tainted run.

These tests use a mock PolicyStore so they assert the wiring (what subjects/
resource reach `check()`, and that a deny raises) without executing real tools.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from continuum.agent.exceptions import ToolAccessDeniedError
from continuum.llm.types import FunctionCall, ToolCall
from continuum.security.policy import PolicyDecision
from continuum.tools.executor import ToolExecutor


def _executor_with_tool(name: str) -> ToolExecutor:
    ex = ToolExecutor.__new__(ToolExecutor)
    # Policy check happens right after this lookup and before any execution,
    # so a placeholder (server, tool) tuple is enough.
    ex.tool_registry = {name: (object(), object())}
    return ex


def _call(name: str) -> ToolCall:
    return ToolCall(id="tc-1", type="function", function=FunctionCall(name=name, arguments="{}"))


class TestDataLabelToolPolicy:
    async def test_labels_are_passed_as_additional_subjects(self):
        ps = MagicMock()
        ps.check.return_value = PolicyDecision(allowed=True, reason="ok")
        ex = _executor_with_tool("send_email")

        # allowed=True → it proceeds past the gate; we only care that the gate
        # was asked the right question, so swallow whatever execution does after.
        with pytest.raises(Exception):  # noqa: B017 - real execution fails on the stub tool
            await ex.execute_tool_call(
                _call("send_email"),
                policy_store=ps,
                subject="agent",
                data_labels={"pii", "financial"},
            )

        # The agent name AND the (sorted) data labels reached the policy check.
        ps.check.assert_called_once_with(["agent", "financial", "pii"], "tool:send_email")

    async def test_deny_by_label_raises_tool_access_denied(self):
        ps = MagicMock()
        ps.check.return_value = PolicyDecision(
            allowed=False, policy_name="no_pii_email", reason="deny"
        )
        ex = _executor_with_tool("send_email")

        with pytest.raises(ToolAccessDeniedError):
            await ex.execute_tool_call(
                _call("send_email"),
                policy_store=ps,
                subject="agent",
                data_labels={"pii"},
            )
        ps.check.assert_called_once_with(["agent", "pii"], "tool:send_email")

    async def test_no_labels_passes_plain_subject_string(self):
        # Backward compatible: no data_labels → subject is the bare agent string.
        ps = MagicMock()
        ps.check.return_value = PolicyDecision(allowed=False, policy_name="p", reason="deny")
        ex = _executor_with_tool("send_email")

        with pytest.raises(ToolAccessDeniedError):
            await ex.execute_tool_call(
                _call("send_email"),
                policy_store=ps,
                subject="agent",
                data_labels=None,
            )
        ps.check.assert_called_once_with("agent", "tool:send_email")
