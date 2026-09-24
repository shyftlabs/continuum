"""A scanner that crashes must not become a scanner that approves (F11).

``input_scanners`` is the only control on the input path that can actually
refuse: it returns ``is_safe=False`` and ``prepare_messages`` raises
``InputBlockedError``. Everything beside it is advisory --
``detect_injection_patterns`` writes a log line and nothing reads the result.

So the scanner's exception path IS the control's failure mode. Today both call
sites swallow it:

    except Exception as e:
        logger.warning("Input scanner %s failed (fail-open): %s", ...)

which means the input reaches the model unscanned, and the only record is a
warning in a log nobody is watching at the time.

WHY THAT IS EXPLOITABLE, NOT MERELY UNTIDY

A scanner is usually a model or a remote call, so MAKING IT THROW is often
cheaper than evading it: oversized input hits a tokenizer or memory limit,
malformed unicode hits a decode error, a hosted classifier times out under load
an attacker can help create. Fail-open turns "crash the scanner" into a complete
bypass of the only thing that can say no.

Worse, it makes scanner quality self-defeating. A real ML classifier has
strictly more ways to fail than a regex, so upgrading the scanner enlarges the
bypass. Fixing this is what makes a better scanner worth installing at all.

WHY NO CONFIG FLAG

The block is gated on ``if ... and agent.config.input_scanners:`` -- an empty
list skips it entirely -- so failing closed changes behaviour ONLY for
deployments that explicitly wired a scanner. Those deployments asked for input
to be scanned; "scanning broke, so I let it through" was never what they asked
for. A flag defaulting to fail-open would protect nobody who did not already go
looking for it.

The escape hatch is better expressed at the scanner anyway: a caller who wants
the old behaviour catches inside their own scanner and returns
``(text, True, None)`` -- three lines, sitting next to the risk being accepted,
visible in review, instead of a global flag far away from it. The raised error
says so, because the person who hits this at 3am needs the fix and not just the
symptom.

THE TWO PATHS DIFFER ON PURPOSE

``prepare_messages`` raises; ``_scan_handoff_payload`` returns a reason string.
That asymmetry is pre-existing and deliberate (handoff_executor.py): the
surrounding runner turns a reason into a failed ``HandoffResult``, where an
escaping exception would crash the whole run instead of failing one transfer.
Fail-closed must respect it, so these tests assert the reason, not a raise. That
a returned reason becomes a clean failure is already covered by
``test_handoff_content_trust.py::test_blocked_payload_fails_the_handoff_cleanly``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from continuum.agent.utils.context_utils import create_run_context
from continuum.exceptions import InputBlockedError


def _agent(**config_kwargs: Any):
    """An agent with the advisory guards off, so only scanners are in play."""
    from continuum.agent.base import BaseAgent
    from continuum.agent.config import AgentConfig, AgentMemoryConfig

    config_kwargs.setdefault("input_sanitization", False)
    config_kwargs.setdefault("injection_detection", False)
    return BaseAgent(
        name="scanned-agent",
        instructions="You are helpful.",
        config=AgentConfig(**config_kwargs),
        memory_config=AgentMemoryConfig(search_memories=False),
    )


def _builder():
    from continuum.agent.execution.message_builder import MessageBuilder

    mem_svc = MagicMock()
    mem_svc.retrieve_memories = AsyncMock(return_value=[])
    sess_svc = MagicMock()
    sess_svc.get_conversation_history = AsyncMock(return_value=[])
    return MessageBuilder(memory_service=mem_svc, session_service=sess_svc)


async def _prepare(agent, text: str = "hello"):
    with patch("continuum.observability.decorators.observe", lambda **kw: lambda f: f):
        return await _builder().prepare_messages(agent, text, create_run_context())


def _exploding(exc: Exception | None = None):
    """A scanner that fails the way a real one does — a model or a remote call."""

    def scanner(text: str) -> tuple[str, bool, str | None]:
        raise exc or RuntimeError("classifier backend unreachable")

    scanner.__name__ = "exploding_scanner"
    return scanner


def _handoff_payload(context: str = "please continue"):
    from continuum.agent.types import HandoffData

    # A real HandoffData, not a mock: _scan_handoff_payload str()s and joins
    # `reason`/`context`, so a mock would let a shape through that production
    # code would never see.
    return HandoffData(
        handoff_id="h1",
        from_agent="agent-a",
        to_agent="agent-b",
        reason="handing over",
        context=context,
        history=[],
    )


def _scan(target_agent) -> str | None:
    from continuum.agent.execution.handoff_executor import HandoffExecutor

    return HandoffExecutor._scan_handoff_payload(
        target_agent, _handoff_payload(), [{"role": "user", "content": "the transferred turn"}]
    )


# ==========================================================================
# The input path — prepare_messages
# ==========================================================================


class TestACrashingScannerBlocksTheInput:
    async def test_a_raising_scanner_does_not_let_the_input_through(self):
        """The finding itself. Today this returns messages and the run proceeds
        with input nothing has scanned."""
        with pytest.raises(InputBlockedError):
            await _prepare(_agent(input_scanners=[_exploding()]))

    async def test_the_error_carries_the_underlying_failure(self):
        """An operator seeing a blocked request needs to know WHICH scanner and
        WHY, or the only actionable signal is a log line they may not have."""
        with pytest.raises(InputBlockedError) as excinfo:
            await _prepare(_agent(input_scanners=[_exploding(ValueError("token limit"))]))

        message = str(excinfo.value)
        assert "exploding_scanner" in message
        assert "token limit" in message

    async def test_the_error_names_the_escape_hatch(self):
        """Failing closed is only defensible if the way back is discoverable at
        the moment it bites. The same reason the Temporal adapter spells out
        'the GLOBAL client needs connecting' rather than just 'not connected'."""
        with pytest.raises(InputBlockedError) as excinfo:
            await _prepare(_agent(input_scanners=[_exploding()]))

        message = str(excinfo.value).lower()
        assert "true" in message, "the message must show the tuple that re-opens the gate"

    async def test_a_later_scanner_never_runs_after_one_crashes(self):
        """Chained scanners pass text along. Continuing past a crash would feed
        the next scanner input the previous one never finished with."""
        reached: list[str] = []

        def second(text: str) -> tuple[str, bool, str | None]:
            reached.append(text)
            return text, True, None

        with pytest.raises(InputBlockedError):
            await _prepare(_agent(input_scanners=[_exploding(), second]))

        assert reached == [], "a scanner ran on input the failed scanner had not cleared"


class TestWhatMustNotChange:
    """Failing closed must not become failing at everything. Only the exception
    path moves; every other outcome is exactly as it was."""

    async def test_an_agent_with_no_scanners_is_untouched(self):
        """The overwhelmingly common case — input_scanners defaults to []. If
        this ever raises, the fix has become an outage."""
        messages, index = await _prepare(_agent())
        assert messages[index]["content"] == "hello"

    async def test_a_healthy_scanner_still_decides(self):
        seen: list[str] = []

        def scanner(text: str) -> tuple[str, bool, str | None]:
            seen.append(text)
            return text, True, None

        messages, index = await _prepare(_agent(input_scanners=[scanner]))
        assert seen == ["hello"]
        assert messages[index]["content"] == "hello"

    async def test_a_refusing_scanner_still_refuses_with_its_own_reason(self):
        """is_safe=False was already blocking. The new path must not swallow or
        rewrite the scanner's own explanation."""

        def blocker(text: str) -> tuple[str, bool, str | None]:
            return text, False, "prompt_injection"

        with pytest.raises(InputBlockedError) as excinfo:
            await _prepare(_agent(input_scanners=[blocker]))
        assert "prompt_injection" in str(excinfo.value)

    async def test_a_scanner_may_still_rewrite_the_input(self):
        """Scanners return sanitized text; redaction must survive the change."""

        def redactor(text: str) -> tuple[str, bool, str | None]:
            return text.replace("hello", "[REDACTED]"), True, None

        messages, index = await _prepare(_agent(input_scanners=[redactor]))
        assert messages[index]["content"] == "[REDACTED]"


# ==========================================================================
# The handoff path — _scan_handoff_payload
# ==========================================================================


class TestACrashingScannerBlocksTheHandoff:
    """Same defect, second call site. One fixed and one missed is
    indistinguishable from working, for whichever path a deployment happens not
    to take."""

    async def test_a_raising_scanner_refuses_the_transfer(self):
        reason = _scan(_agent(input_scanners=[_exploding()]))
        assert reason is not None, "a crashed scanner let the payload through"

    async def test_it_returns_a_reason_rather_than_raising(self):
        """The pre-existing contract: the caller turns a reason into a failed
        HandoffResult. An exception escaping here crashes the entire run instead
        of failing one transfer."""
        agent = _agent(input_scanners=[_exploding()])
        try:
            reason = _scan(agent)
        except Exception as e:  # noqa: BLE001 - the assertion is that this cannot happen
            pytest.fail(
                f"_scan_handoff_payload raised {type(e).__name__} instead of returning: {e}"
            )
        assert isinstance(reason, str)

    async def test_the_reason_identifies_the_failed_scanner(self):
        reason = _scan(_agent(input_scanners=[_exploding(ValueError("token limit"))])) or ""
        assert "exploding_scanner" in reason
        assert "token limit" in reason

    async def test_a_healthy_scanner_still_allows_the_transfer(self):
        def scanner(text: str) -> tuple[str, bool, str | None]:
            return text, True, None

        assert _scan(_agent(input_scanners=[scanner])) is None

    async def test_a_refusing_scanner_still_refuses(self):
        def blocker(text: str) -> tuple[str, bool, str | None]:
            return text, False, "prompt_injection"

        assert _scan(_agent(input_scanners=[blocker])) == "prompt_injection"

    async def test_an_agent_with_no_scanners_still_accepts_handoffs(self):
        assert _scan(_agent()) is None
