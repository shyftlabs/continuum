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

from continuum.agent import BaseAgent
from continuum.config import Settings, settings
from continuum.logging import PromptContentFilter, log_content, setup_logging

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


# ── declared content is withheld, and nothing else is ─────────────────────────


class TestDeclaredContentIsWithheld:
    def test_it_does_not_survive_in_any_form(self, content_logging_off):
        rendered = _filtered(
            _record("===== FINAL PROMPT [%s] =====\n%s", "clinic", log_content(PHI * 3))
        )
        assert PHI not in rendered
        assert "PATIENT" not in rendered, "a prefix of the record is still the record"

    def test_the_developer_s_structure_survives(self, content_logging_off):
        """The line has to stay useful: which agent, which event."""
        rendered = _filtered(
            _record("===== FINAL PROMPT [%s] =====\n%s", "clinic", log_content(PHI * 3))
        )
        assert "FINAL PROMPT" in rendered
        assert "clinic" in rendered

    def test_it_says_how_much_was_withheld(self, content_logging_off):
        blob = PHI * 3
        rendered = _filtered(_record("prompt: %s", log_content(blob)))
        assert f"<{len(blob)} chars>" in rendered

    def test_the_record_still_renders(self, content_logging_off):
        """A bare %s left in the output would mean an argument was dropped
        rather than replaced."""
        rendered = _filtered(_record("a=%s b=%s", log_content(PHI * 3), log_content(PHI * 3)))
        assert "%s" not in rendered


class TestUndeclaredArgumentsAreUntouched:
    """A length threshold used to sit here, withholding any argument over 64
    characters. It was wrong in both directions -- 46 characters of PHI passed
    it, a 65-character pasteable `continuum mcp diff` command did not -- and was
    removed. These pin its absence, because length is not a property that
    distinguishes a medical note from a file path.
    """

    def test_a_short_label_is_untouched(self, content_logging_off):
        rendered = _filtered(_record("Gateway selected model: %s", "claude-opus-5"))
        assert rendered == "Gateway selected model: claude-opus-5"

    def test_non_string_arguments_are_untouched(self, content_logging_off):
        rendered = _filtered(_record("agent=%s message_count=%d", "clinic", 7))
        assert rendered == "agent=clinic message_count=7"

    def test_a_long_label_survives_whole(self, content_logging_off):
        """The F3 case, at the length that used to break it."""
        command = "continuum mcp diff srv --pins /etc/continuum/tool-trust/pins.json"
        assert len(command) > 64
        assert _filtered(_record("Review with `%s`.", command)) == f"Review with `{command}`."

    def test_an_undeclared_value_leaks(self, content_logging_off):
        """Stated rather than implied: forgetting log_content() leaks, and the
        thing that catches that is tests/unit/test_log_canary.py, not this
        filter."""
        assert PHI in _filtered(_record("memory: %s", PHI))

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


# ── declared content, at any length ───────────────────────────────────────────


class TestDeclaredContent:
    """PHI is 46 characters. So is a session id and a model name. No rule based
    on the value itself can separate them, which is why the call site declares."""

    def test_short_content_is_withheld_when_declared(self, content_logging_off):
        assert PHI not in _filtered(_record("memory: %s", log_content(PHI)))

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

    def test_a_child_logger_reaches_the_filter(self, content_logging_on, captured_log):
        """Every module logs through continuum.<module>, whose records reach the
        parent's handlers by propagation. Propagation runs the ancestors'
        *handlers* and skips their filters, so a filter on the logger would never
        see these records.

        Asserted with content logging ON, because that is the direction the
        filter is responsible for: withholding happens in _Content.__str__ and
        would pass even with no filter installed at all, so it proves nothing
        about wiring. Revealing only happens if the filter actually ran.
        """
        logging.getLogger("continuum.agent.execution.message_builder").info(
            "prompt: %s", log_content(PHI)
        )
        assert PHI in captured_log.getvalue()


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


# ── the tool list is the system's own fact, and prints ────────────────────────

_SHOP_TOOL = {
    "type": "function",
    "function": {
        "name": "shop__search_products",
        "description": "Search the catalogue",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
    },
}


@pytest.fixture
def info_records(content_logging_off):
    """Every INFO+ record under continuum, as (level, rendered) pairs."""
    records: list[tuple[int, str]] = []

    class Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append((record.levelno, record.getMessage()))

    root = logging.getLogger("continuum")
    handler, level = Collector(), root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        yield records
    finally:
        root.removeHandler(handler)
        root.setLevel(level)


def _tools_line(records):
    found = [(lvl, msg) for lvl, msg in records if "===== TOOLS [" in msg]
    assert found, f"no TOOLS line was emitted: {[m for _, m in records]}"
    return found[0]


@pytest.mark.asyncio
class TestTheToolListPrints:
    """The TOOLS line lists each tool's name and parameter *schema* -- the shape
    of its arguments, defined by the developer or the MCP server before any user
    arrived. It is the system's own fact, so by the rule it prints. It was
    wrapped in log_content() in the first commit of this work, before that rule
    was settled, because it is emitted beside FINAL PROMPT and "everything we
    send the model" was treated as content. The prompt carries the user's input;
    the tool list does not.

    Kept at INFO deliberately: whether it is noisy is a separate question from
    whether it is the user's data, and only the second is this filter's job."""

    async def test_the_agent_s_tool_schemas_print_by_default(self, info_records):
        from continuum.agent.execution.message_builder import MessageBuilder

        agent = BaseAgent(name="shop", instructions="hi", tools=[_SHOP_TOOL])
        await MessageBuilder().prepare_messages(
            agent=agent, input="find socks", context=_run_context()
        )
        _, line = _tools_line(info_records)
        assert "shop__search_products" in line
        assert "'query'" in line, "the parameter schema is what was being withheld"
        assert "chars>" not in line

    async def test_it_stays_at_info(self, info_records):
        from continuum.agent.execution.message_builder import MessageBuilder

        agent = BaseAgent(name="shop", instructions="hi", tools=[_SHOP_TOOL])
        await MessageBuilder().prepare_messages(
            agent=agent, input="find socks", context=_run_context()
        )
        level, _ = _tools_line(info_records)
        assert level == logging.INFO

    async def test_the_prompt_beside_it_is_still_withheld(self, info_records):
        """The neighbouring line carries the user's input and must not follow."""
        from continuum.agent.execution.message_builder import MessageBuilder

        agent = BaseAgent(name="shop", instructions="hi", tools=[_SHOP_TOOL])
        await MessageBuilder().prepare_messages(agent=agent, input=PHI, context=_run_context())
        assert not any(PHI in msg for _, msg in info_records)

    async def test_a_handoff_target_s_tool_schemas_print_too(self, info_records):
        """The same line, emitted for the agent a handoff lands on."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from continuum.agent.execution.handoff_executor import HandoffExecutor
        from continuum.agent.handoff.manager import HandoffManager
        from continuum.agent.types import (
            HANDOFF_TOOL_PREFIX,
            AgentResponse,
            ResponseStatus,
            RunState,
        )
        from continuum.agent.utils.context_utils import create_run_context

        hm = MagicMock(spec=HandoffManager)
        hm._max_depth = 10
        hm.detect_cycle = MagicMock(return_value=False)
        hm.prepare_handoff = AsyncMock(
            return_value=MagicMock(handoff_id="h1", to_dict=MagicMock(return_value={}))
        )
        hm.build_handoff_messages = MagicMock(return_value=[{"role": "user", "content": "go"}])
        hm.trace_handoff = AsyncMock()

        async def fake_loop(agent, messages, context, run_state):
            return AgentResponse(content="ok", agent_name=agent.name, status=ResponseStatus.SUCCESS)

        inner = MagicMock()
        inner.execute_loop = fake_loop
        executor = HandoffExecutor(handoff_manager=hm, agent_registry={}, executor=inner)
        executor.register_agent(BaseAgent(name="shop", instructions="hi", tools=[_SHOP_TOOL]))

        call = MagicMock()
        call.function.name = f"{HANDOFF_TOOL_PREFIX}shop"
        call.function.arguments = '{"reason": "test"}'
        call.id = "tc-1"
        state = RunState(run_id="run-1")
        state.push_agent("front")

        with patch("continuum.observability.decorators.observe", lambda **kw: lambda f: f):
            await executor.execute_handoff(
                BaseAgent(name="front", instructions="hi"),
                "shop",
                call,
                [],
                create_run_context(session_id="sess-1"),
                state,
            )
        level, line = _tools_line(info_records)
        assert "shop__search_products" in line
        assert "'query'" in line
        assert level == logging.INFO


def _run_context():
    from continuum.agent.types import RunContext

    return RunContext(run_id="run-tools")


# ── the documentation must not drift from the code ────────────────────────────


class TestTheDocsMatchTheCode:
    """docs/installation.md described the 64-character backstop for a while
    after it was deleted, so it promised a guarantee that no longer existed.
    These pin the claims that section makes."""

    @staticmethod
    def _doc() -> str:
        from pathlib import Path

        return (Path(__file__).resolve().parents[2] / "docs" / "installation.md").read_text()

    @staticmethod
    def _pyproject() -> str:
        from pathlib import Path

        return (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()

    def test_it_documents_the_setting_that_exists(self):
        assert "LOG_PROMPT_CONTENT" in self._doc()
        assert "log_prompt_content" in Settings.model_fields

    def test_it_names_both_wrappers_and_both_are_importable(self):
        from continuum import logging as continuum_logging

        doc = self._doc()
        for name in ("log_content()", "log_id()"):
            assert name in doc, name
            assert hasattr(continuum_logging, name.rstrip("()"))

    def test_it_does_not_still_promise_the_deleted_backstop(self):
        """The specific drift that happened. 'longer than 64 characters' was a
        guarantee the code stopped making."""
        assert "longer than 64 characters" not in self._doc()
        assert not hasattr(__import__("continuum.logging", fromlist=["x"]), "ARGUMENT_LIMIT")

    def test_the_G004_claim_is_true(self):
        """The doc says the rule is enabled for all of src/ with no exemption.
        An exemption added later would make that a lie."""
        pyproject = self._pyproject()
        assert "no\nexemption" in self._doc() or "no exemption" in self._doc()
        assert '"G004"' in pyproject, "G004 is not selected"
        exempted = [
            line for line in pyproject.splitlines() if line.startswith('"src/') and "G004" in line
        ]
        assert exempted == [], f"src/ carries a G004 exemption: {exempted}"

    def test_no_f_string_logging_call_remains_in_src(self):
        """The other half of that claim, checked against the code rather than
        against the lint config."""
        import ast
        from pathlib import Path

        offenders = []
        for path in (Path(__file__).resolve().parents[2] / "src" / "continuum").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                    continue
                if node.func.attr not in ("info", "debug", "warning", "error", "critical"):
                    continue
                if not (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id in ("logger", "_logger")
                ):
                    continue
                if node.args and isinstance(node.args[0], ast.JoinedStr):
                    offenders.append(f"{path.name}:{node.lineno}")
        assert offenders == [], offenders
