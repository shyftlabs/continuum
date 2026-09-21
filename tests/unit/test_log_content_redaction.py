"""Prompt content must not reach the log by default.

Continuum assembles a prompt from the system instructions, retrieved memories,
session history, RAG context and the user's input, then logs the whole thing at
``INFO`` -- the shipped default level -- in ``message_builder.prepare_messages``.
Neither formatter in ``continuum.logging`` redacts anything, so in production
(``JSONFormatter``) that lands in the log aggregator verbatim. The existing
``data_redaction`` module guards *telemetry*, not this path.

The fix is a seam rather than a sweep of the call sites: one filter on the
handlers every ``continuum.*`` logger reaches, so code that has not been written
yet is covered too.

Its contract is narrow, and the tests below pin both halves:

  protected      ``logger.info("FINAL PROMPT [%s]\\n%s", name, prompt)`` keeps
                 the format string and the values apart until the formatter
                 runs. The literal is the developer's structure; the values are
                 the data. A long value is *replaced*, not shortened -- a prefix
                 of a patient record is still a patient record.

  not protected  An f-string site has already collapsed the two by the time the
                 record exists. Capping such a message by length was tried and
                 removed: it cannot tell a prompt dump from a thorough operator
                 message, and it destroyed the pasteable ``continuum mcp diff``
                 command in the tool-trust warnings. A site logs content safely
                 by passing it as an argument, which is a visible, greppable act.

``exc_info`` is deliberately untouched. The leak is the exception *text*
interpolated into a message; the traceback is a different field on the record,
carries no locals, and is the only thing that says where a failure happened.
"""

from __future__ import annotations

import io
import logging

import pytest

from continuum.config import Settings, settings
from continuum.logging import (
    ARGUMENT_LIMIT,
    PromptContentFilter,
    log_content,
    setup_logging,
)

PHI = "PATIENT-SSN-123-45-6789-DIAGNOSIS-HIV-POSITIVE"


def _record(msg: str, *args: object, **kwargs: object) -> logging.LogRecord:
    """A record as ``logger.info(msg, *args)`` would produce it."""
    return logging.LogRecord(
        name="continuum.probe",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=kwargs.get("exc_info"),  # type: ignore[arg-type]
    )


def _filtered(record: logging.LogRecord) -> str:
    """Run the filter and return what a formatter would render."""
    assert PromptContentFilter().filter(record) is True, "the line must still be emitted"
    return record.getMessage()


@pytest.fixture
def content_logging_off(monkeypatch):
    monkeypatch.setattr(settings, "log_prompt_content", False)


@pytest.fixture
def content_logging_on(monkeypatch):
    monkeypatch.setattr(settings, "log_prompt_content", True)


# ── the shipped default ───────────────────────────────────────────────────────


class TestTheShippedDefault:
    def test_content_logging_is_off_by_default(self):
        """Asserted against the field default, not the live object: a developer
        .env that sets LOG_PROMPT_CONTENT=true would otherwise make this pass or
        fail on whose machine it ran, which is the trap the session settings hit."""
        assert Settings.model_fields["log_prompt_content"].default is False


# ── the protected half: arguments are replaced, not shortened ─────────────────


class TestArgumentsAreElided:
    def test_a_long_argument_does_not_survive_in_any_form(self, content_logging_off):
        rendered = _filtered(_record("===== FINAL PROMPT [%s] =====\n%s", "clinic", PHI * 3))
        assert PHI not in rendered
        assert "PATIENT" not in rendered, "a prefix of the record is still the record"

    def test_the_developer_s_structure_survives(self, content_logging_off):
        """The line has to stay useful: which agent, which event."""
        rendered = _filtered(_record("===== FINAL PROMPT [%s] =====\n%s", "clinic", PHI * 3))
        assert "FINAL PROMPT" in rendered
        assert "clinic" in rendered

    def test_the_elision_says_how_much_was_withheld(self, content_logging_off):
        blob = PHI * 3
        rendered = _filtered(_record("prompt: %s", blob))
        assert f"<{len(blob)} chars>" in rendered

    def test_a_short_argument_is_untouched(self, content_logging_off):
        rendered = _filtered(_record("Gateway selected model: %s", "claude-opus-5"))
        assert rendered == "Gateway selected model: claude-opus-5"

    def test_non_string_arguments_are_untouched(self, content_logging_off):
        rendered = _filtered(_record("agent=%s message_count=%d", "clinic", 7))
        assert rendered == "agent=clinic message_count=7"

    def test_an_argument_at_the_limit_is_kept(self, content_logging_off):
        value = "x" * ARGUMENT_LIMIT
        assert _filtered(_record("v=%s", value)) == f"v={value}"

    def test_one_past_the_limit_goes(self, content_logging_off):
        value = "x" * (ARGUMENT_LIMIT + 1)
        assert _filtered(_record("v=%s", value)) == f"v=<{ARGUMENT_LIMIT + 1} chars>"

    def test_the_record_still_renders_after_the_swap(self, content_logging_off):
        """If args were dropped instead of replaced, getMessage() would raise or
        leave a bare %s in the output."""
        rendered = _filtered(_record("a=%s b=%s", PHI * 3, PHI * 3))
        assert "%s" not in rendered

    def test_mapping_style_args_are_left_alone(self, content_logging_off):
        """logging also accepts a single dict for %(name)s interpolation. It is
        not a shape this codebase uses, and mishandling it would crash the line."""
        record = _record("v=%(v)s", {"v": "short"})
        assert _filtered(record) == "v=short"


# ── the unprotected half, stated so nobody assumes coverage ───────────────────


class TestAPreformattedMessageIsNotProtected:
    """Pinned deliberately. A filter cannot separate structure from data in a
    string that was already interpolated, and the length cap that was tried
    instead is recorded below as the reason not to try it again."""

    def test_an_f_string_site_still_leaks(self, content_logging_off):
        rendered = _filtered(_record(f"TOOL RESULT: charge_card -> {PHI}"))
        assert PHI in rendered, (
            "If this now passes, someone added message-level redaction. Read the "
            "next test before keeping it."
        )

    def test_a_thorough_operator_message_survives_intact(self, content_logging_off):
        """Why the length cap was removed. This is the real tool-trust warning
        (F3), whose whole point is that the command can be pasted. A 200-char cap
        cut the path off mid-word and broke four tests in
        tests/unit/tools/test_tool_trust_enforcement.py."""
        line = (
            "MCP server 'srv': ['sneaked_in'] were dropped -- they are not in the "
            "approved catalogue. Review with `continuum mcp diff srv --pins "
            "/etc/continuum/tool-trust/pins.json` and pin what you accept."
        )
        assert _filtered(_record(line)) == line

    def test_the_same_content_is_protected_once_it_is_declared(self, content_logging_off):
        """The migration for a leaking site: one line, no truncation logic, and
        the full value is still there when an operator asks for it."""
        rendered = _filtered(_record("TOOL RESULT: %s -> %s", "charge_card", log_content(PHI)))
        assert PHI not in rendered
        assert "charge_card" in rendered


# ── declared content, which length cannot catch ───────────────────────────────


class TestDeclaredContent:
    """PHI is 46 characters. So is a session id and a model name. The length
    backstop catches prompt dumps; only the call site can catch a short note."""

    def test_short_content_is_withheld_when_declared(self, content_logging_off):
        assert PHI not in _filtered(_record("memory: %s", log_content(PHI)))

    def test_the_same_value_undeclared_slips_past_the_backstop(self, content_logging_off):
        """Why log_content() exists rather than a lower ARGUMENT_LIMIT: dropping
        the threshold under 46 would also elide session ids and model names,
        which are exactly what a log line is for."""
        assert len(PHI) < ARGUMENT_LIMIT
        assert PHI in _filtered(_record("memory: %s", PHI))

    def test_it_still_says_how_much_was_withheld(self, content_logging_off):
        assert f"<{len(PHI)} chars>" in _filtered(_record("memory: %s", log_content(PHI)))

    def test_it_is_revealed_when_the_operator_asks(self, content_logging_on):
        assert _filtered(_record("memory: %s", log_content(PHI))) == f"memory: {PHI}"

    def test_it_fails_closed_with_no_filter_at_all(self):
        """A handler this SDK did not install -- someone's own, or a bare
        basicConfig -- never runs the filter. The wrapper must not print the
        value on its own."""
        assert PHI not in str(log_content(PHI))
        assert PHI not in repr(log_content(PHI))
        # getMessage() is the interpolation logging performs, run here with no
        # filter in front of it at all.
        assert PHI not in _record("memory: %s", log_content(PHI)).getMessage()

    def test_a_non_string_value_is_handled(self, content_logging_off):
        """Tool arguments arrive as dicts."""
        rendered = _filtered(_record("args: %s", log_content({"card": "4111111111111111"})))
        assert "4111111111111111" not in rendered


# ── the switch ────────────────────────────────────────────────────────────────


class TestTheSwitch:
    def test_content_passes_through_when_it_is_on(self, content_logging_on):
        blob = PHI * 3
        assert _filtered(_record("prompt: %s", blob)) == f"prompt: {blob}"

    def test_an_argument_is_whole_when_it_is_on(self, content_logging_on):
        """Debugging "why did the agent ignore my memory?" needs the real text,
        not a length. That is what the switch is for."""
        assert _filtered(_record("TOOL RESULT: %s -> %s", "charge_card", PHI)) == (
            f"TOOL RESULT: charge_card -> {PHI}"
        )


# ── the traceback is a different field, and stays ─────────────────────────────


class TestTracebacksSurvive:
    def test_exc_info_is_not_touched(self, content_logging_off):
        """PR #100 dropped exc_info=True alongside the leaky f-string. The leak
        was the interpolated text; the traceback is what tells you where."""
        try:
            raise TimeoutError("target agent timed out")
        except TimeoutError:
            import sys

            record = _record(
                "Failed to execute target agent: %s", "researcher", exc_info=sys.exc_info()
            )

        PromptContentFilter().filter(record)

        assert record.exc_info is not None
        assert record.exc_info[0] is TimeoutError
        assert "Traceback" in logging.Formatter().formatException(record.exc_info)


# ── the filter has to actually be attached ────────────────────────────────────


@pytest.fixture
def captured_log():
    """Real setup_logging wiring, with the console handler's stream swapped.

    Asserting against our own hand-built handler would prove the class works and
    say nothing about whether the SDK installs it.
    """
    root = logging.getLogger("continuum")
    saved_handlers, saved_level = root.handlers[:], root.level
    setup_logging(level="INFO", json_format=False, enable_langfuse_handler=False)
    buffer = io.StringIO()
    root.handlers[0].stream = buffer  # type: ignore[attr-defined]
    try:
        yield buffer
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


class TestTheFilterIsWired:
    def test_every_handler_carries_it(self):
        root = logging.getLogger("continuum")
        saved = root.handlers[:]
        setup_logging(level="INFO", json_format=False, enable_langfuse_handler=True)
        try:
            assert root.handlers, "setup_logging installed no handlers"
            for handler in root.handlers:
                assert any(isinstance(f, PromptContentFilter) for f in handler.filters), (
                    f"{type(handler).__name__} would emit unredacted content"
                )
        finally:
            root.handlers[:] = saved

    def test_a_child_logger_is_covered(self, content_logging_off, captured_log):
        """Every module logs through continuum.<module>, whose records reach the
        parent's handlers by propagation. A filter on the *logger* would be
        skipped for those; only a handler filter sees them."""
        logging.getLogger("continuum.agent.execution.message_builder").info("prompt: %s", PHI * 3)
        assert PHI not in captured_log.getvalue()


# ── the leak this was written for ─────────────────────────────────────────────


@pytest.mark.asyncio
class TestTheRealLeak:
    async def test_an_assembled_prompt_does_not_reach_the_log(
        self, content_logging_off, captured_log
    ):
        """Run the real path at the real default level. Before the filter this
        printed the FINAL PROMPT block with the input in it."""
        from continuum.agent import BaseAgent
        from continuum.agent.execution.message_builder import MessageBuilder
        from continuum.agent.types import RunContext

        agent = BaseAgent(name="clinic", instructions="You are a triage bot.")
        await MessageBuilder().prepare_messages(
            agent=agent, input=PHI, context=RunContext(run_id="run-abc-123")
        )

        out = captured_log.getvalue()
        assert out, "nothing was logged — the test is not exercising the path"
        assert PHI not in out
        assert "FINAL PROMPT" in out, "the diagnostic itself should survive, only its content goes"
