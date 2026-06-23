"""@observe must treat expected access-control denials (PolicyDeniedError) as
governance events, not failures: they are NOT escalated to error reporting (no
high-severity Langfuse error trace / no alert), while every other exception is.
The exception still propagates either way (control flow unchanged)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from continuum.agent.exceptions import (
    MemoryAccessDeniedError,
    ModelAccessDeniedError,
    ToolAccessDeniedError,
)
from continuum.observability.decorators import observe
from continuum.observability.error_reporter import report_error


class TestObservePolicyDeniedNotReported:
    async def test_async_policy_denial_is_not_reported(self):
        @observe(name="llm_chat")
        async def run():
            raise ModelAccessDeniedError(model="gpt-4o", policy_name="phi-no-cloud-model")

        with patch("continuum.observability.error_reporter.report_error") as rep:
            with pytest.raises(ModelAccessDeniedError):  # still propagates
                await run()
        rep.assert_not_called()  # expected denial → NOT escalated to error reporting

    async def test_async_normal_error_is_reported(self):
        @observe(name="llm_chat")
        async def run():
            raise ValueError("boom")

        with patch("continuum.observability.error_reporter.report_error") as rep:
            with pytest.raises(ValueError):
                await run()
        rep.assert_called_once()  # real failure → reported as before

    def test_sync_policy_denial_is_not_reported(self):
        @observe(name="tool.send_referral_email")
        def run():
            raise ToolAccessDeniedError(tool_name="send_referral_email", policy_name="phi-no-exfil")

        with patch("continuum.observability.error_reporter.report_error") as rep:
            with pytest.raises(ToolAccessDeniedError):
                run()
        rep.assert_not_called()


class TestReportErrorChokepoint:
    """report_error() itself is the single chokepoint: it must drop policy
    denials regardless of which call site invoked it (the agent-trace decorator
    has several), so no high-severity error trace is ever created for them."""

    def test_policy_denied_not_forwarded_to_reporter(self):
        with patch("continuum.observability.error_reporter.get_error_reporter") as gr:
            reporter = MagicMock()
            gr.return_value = reporter
            report_error(MemoryAccessDeniedError(operation="write", policy_name="phi-never-persisted"))
            reporter.report.assert_not_called()

    def test_normal_error_is_forwarded_to_reporter(self):
        with patch("continuum.observability.error_reporter.get_error_reporter") as gr:
            reporter = MagicMock()
            gr.return_value = reporter
            report_error(ValueError("boom"))
            reporter.report.assert_called_once()
