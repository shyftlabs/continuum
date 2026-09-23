"""The user's identity must not reach the log either.

``log_content()`` answers "the user's *words*". This answers the other half:
their *identity*. Two reasons it needs a different mechanism.

First, it is not optional data. On shipped defaults a session id is derived in
plaintext from the user id::

    compute_session_id(user_id="alice@clinic.example", conversation_id="conv-1")
      → "c:conv-1:u:alice@clinic.example"

So ``logger.debug("... session_id=%s", session_id)`` writes an email address to
the aggregator. ``SESSION_HASH_IDS=true`` fixes that at the source and is the
better answer where it can be set; this covers the deployments that cannot.

Second, ``<31 chars>`` is the wrong redaction for an identifier. Every session
renders identically, so an operator can no longer tell whether two lines belong
to the same user, and correlating an incident becomes impossible. A *stable
pseudonym* withholds the identity and keeps the grouping::

    s#4f2a9c1e...          the same id, every time, unlinkable to the person

There is also a channel no call site can reach. ``llm/callbacks.py`` publishes
``user_id`` and ``session_id`` into the logging context, and ``JSONFormatter``
stamps the context onto *every* structured line. That data never passes through
``record.args``, so ``PromptContentFilter`` cannot see it. The formatters
therefore read ``_context_for_output()`` instead of ``get_log_context()``.

``get_log_context()`` itself is deliberately unchanged: ``llm/callbacks.py`` and
``observability/error_reporter.py`` read it for correlation and error
attribution and need the real values. Redact at the exit, not at the source --
the same reasoning that puts the filter on handlers rather than loggers. The
regression test for that is at the bottom and is the one that matters most.
"""

from __future__ import annotations

import io
import json
import logging

import pytest

from continuum.config import settings
from continuum.logging import (
    LogContext,
    _context_for_output,
    get_log_context,
    log_content,
    log_id,
    setup_logging,
)

SECRET = "0123456789abcdef" * 4  # 64 chars, passes the weak-secret guard
ALICE = "c:conv-1:u:alice@clinic.example"
BOB = "c:conv-1:u:bob@clinic.example"


def _record(msg: str, *args: object) -> logging.LogRecord:
    return logging.LogRecord("continuum.probe", logging.INFO, __file__, 1, msg, args, None)


def _filtered(record: logging.LogRecord) -> str:
    from continuum.logging import PromptContentFilter

    assert PromptContentFilter().filter(record) is True
    return record.getMessage()


@pytest.fixture
def keyed(monkeypatch):
    """A deployment with a secret configured, content logging off."""
    monkeypatch.setattr(settings, "log_prompt_content", False)
    monkeypatch.setattr(settings, "session_id_secret", SECRET)


@pytest.fixture
def unkeyed(monkeypatch):
    monkeypatch.setattr(settings, "log_prompt_content", False)
    monkeypatch.setattr(settings, "session_id_secret", None)


@pytest.fixture
def revealed(monkeypatch):
    monkeypatch.setattr(settings, "log_prompt_content", True)
    monkeypatch.setattr(settings, "session_id_secret", SECRET)


# ── the pseudonym ─────────────────────────────────────────────────────────────


class TestPseudonym:
    def test_the_identity_does_not_survive(self, keyed):
        rendered = _filtered(_record("session=%s", log_id(ALICE)))
        assert ALICE not in rendered
        assert "alice" not in rendered
        assert "clinic.example" not in rendered

    def test_it_is_stable_across_calls(self, keyed):
        """Without this the whole point is lost: an operator has to be able to
        group two lines from the same session."""
        first = _filtered(_record("s=%s", log_id(ALICE)))
        second = _filtered(_record("s=%s", log_id(ALICE)))
        assert first == second

    def test_two_identities_differ(self, keyed):
        assert _filtered(_record("s=%s", log_id(ALICE))) != _filtered(_record("s=%s", log_id(BOB)))

    def test_it_is_marked_so_a_reader_knows_what_it_is(self, keyed):
        assert _filtered(_record("s=%s", log_id(ALICE))).startswith("s=id#")

    def test_a_different_secret_gives_a_different_pseudonym(self, monkeypatch, keyed):
        mine = _filtered(_record("s=%s", log_id(ALICE)))
        monkeypatch.setattr(settings, "session_id_secret", "f" * 64)
        assert _filtered(_record("s=%s", log_id(ALICE))) != mine

    def test_it_is_revealed_when_the_operator_asks(self, revealed):
        assert _filtered(_record("s=%s", log_id(ALICE))) == f"s={ALICE}"

    def test_a_non_string_value_is_handled(self, keyed):
        assert "id#" in _filtered(_record("s=%s", log_id(12345)))

    def test_none_is_not_pseudonymised(self, keyed):
        """A missing id is not a secret, and 'id#a1b2' would imply one existed."""
        assert _filtered(_record("s=%s", log_id(None))) == "s=None"


class TestWithoutASecret:
    """An unkeyed hash of an email falls to a word list -- the known-plaintext
    problem PR #98 was about. identity.py already states the rule: a control that
    reports 'enabled' while doing nothing is worse than one honestly off. So with
    no secret configured the identity is withheld outright, and correlation is
    the thing that degrades."""

    def test_it_withholds_rather_than_hashing_weakly(self, unkeyed):
        rendered = _filtered(_record("s=%s", log_id(ALICE)))
        assert ALICE not in rendered
        assert "id#" not in rendered
        assert f"<{len(ALICE)} chars>" in rendered

    def test_it_is_still_revealed_when_asked(self, monkeypatch, unkeyed):
        monkeypatch.setattr(settings, "log_prompt_content", True)
        assert _filtered(_record("s=%s", log_id(ALICE))) == f"s={ALICE}"


class TestFailsClosed:
    """A handler this SDK did not install never runs the filter."""

    def test_str_and_repr_withhold(self, keyed):
        assert ALICE not in str(log_id(ALICE))
        assert ALICE not in repr(log_id(ALICE))

    def test_interpolation_without_a_filter_withholds(self, keyed):
        assert ALICE not in _record("s=%s", log_id(ALICE)).getMessage()


# ── the context channel ───────────────────────────────────────────────────────


@pytest.fixture
def captured_json():
    """Real setup_logging, production JSON formatter, stream swapped."""
    root = logging.getLogger("continuum")
    saved, level = root.handlers[:], root.level
    setup_logging(level="INFO", json_format=True, enable_langfuse_handler=False)
    buffer = io.StringIO()
    root.handlers[0].stream = buffer  # type: ignore[attr-defined]
    try:
        yield buffer
    finally:
        root.handlers[:] = saved
        root.setLevel(level)


def _lines(buffer: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in buffer.getvalue().strip().split("\n") if line.strip()]


class TestContextFieldsAreNotStamped:
    """JSONFormatter copies the log context onto every line. No call-site edit
    can reach that, because the data never passes through record.args."""

    def test_the_user_id_does_not_reach_the_line(self, keyed, captured_json):
        with LogContext(trace_id="t-1", user_id="alice@clinic.example", session_id=ALICE):
            logging.getLogger("continuum.probe").info("anything at all")
        assert "alice@clinic.example" not in captured_json.getvalue()

    def test_the_session_id_does_not_reach_the_line(self, keyed, captured_json):
        with LogContext(trace_id="t-1", user_id="alice@clinic.example", session_id=ALICE):
            logging.getLogger("continuum.probe").info("anything at all")
        assert ALICE not in captured_json.getvalue()

    def test_the_fields_are_still_there_as_pseudonyms(self, keyed, captured_json):
        """Withheld, not dropped: an operator still gets a value to group by."""
        with LogContext(trace_id="t-1", user_id="alice@clinic.example", session_id=ALICE):
            logging.getLogger("continuum.probe").info("anything at all")
        line = _lines(captured_json)[0]
        assert line["user_id"].startswith("id#")
        assert line["session_id"].startswith("id#")

    def test_the_trace_id_survives_whole(self, keyed, captured_json):
        """Continuum generates it, it is derived from nobody, and it is the last
        correlation thread. Over-redacting here costs everything and buys nothing."""
        with LogContext(trace_id="trace-abc-123", user_id="alice@clinic.example"):
            logging.getLogger("continuum.probe").info("anything at all")
        assert _lines(captured_json)[0]["trace_id"] == "trace-abc-123"

    def test_it_is_revealed_when_the_operator_asks(self, revealed, captured_json):
        with LogContext(trace_id="t-1", user_id="alice@clinic.example"):
            logging.getLogger("continuum.probe").info("anything at all")
        assert _lines(captured_json)[0]["user_id"] == "alice@clinic.example"


class TestTheDevelopmentFormatterToo:
    """It prints `user=...` in its context prefix, so it leaks the same value by
    a different route. The rule follows the setting, not the formatter -- JSON
    format runs locally too."""

    def test_the_user_id_does_not_reach_the_line(self, keyed):
        root = logging.getLogger("continuum")
        saved, level = root.handlers[:], root.level
        setup_logging(level="INFO", json_format=False, enable_langfuse_handler=False)
        buffer = io.StringIO()
        root.handlers[0].stream = buffer  # type: ignore[attr-defined]
        try:
            with LogContext(trace_id="t-1", user_id="alice@clinic.example"):
                logging.getLogger("continuum.probe").info("anything at all")
            assert "alice@clinic.example" not in buffer.getvalue()
            assert "id#" in buffer.getvalue()
        finally:
            root.handlers[:] = saved
            root.setLevel(level)


class TestOneMechanismForBoth:
    """A session must render the same whether it arrives as a context field or as
    a call-site argument, or the two cannot be joined in a dashboard."""

    def test_the_context_and_the_argument_agree(self, keyed):
        from_argument = _filtered(_record("%s", log_id(ALICE))).strip()
        with LogContext(session_id=ALICE):
            from_context = _context_for_output()["session_id"]
        assert from_argument == from_context


# ── the regression that matters most ──────────────────────────────────────────


class TestTheSourceIsUntouched:
    """llm/callbacks.py correlates traces from this, and
    observability/error_reporter.py attributes errors from it. Pseudonymising at
    the source would corrupt both. The redaction belongs at the exit."""

    def test_get_log_context_still_returns_the_real_values(self, keyed):
        with LogContext(trace_id="t-1", user_id="alice@clinic.example", session_id=ALICE):
            context = get_log_context()
        assert context["user_id"] == "alice@clinic.example"
        assert context["session_id"] == ALICE

    def test_only_the_output_view_is_pseudonymised(self, keyed):
        with LogContext(user_id="alice@clinic.example"):
            assert get_log_context()["user_id"] == "alice@clinic.example"
            assert _context_for_output()["user_id"].startswith("id#")


# ── the error reporter's own exit ─────────────────────────────────────────────


class TestErrorReporterDoesNotShipIdentity:
    """An error raised outside a traced run makes a NEW Langfuse trace, stamped
    with user_id and session_id straight from the log context. Errors should stay
    attributable, but the person should not egress to a third party to do it.

    Only this branch: inside a traced run the report is an event on the existing
    trace and neither field is passed.
    """

    def _report(self, monkeypatch):
        from unittest.mock import MagicMock

        from continuum.observability.error_reporter import ErrorReporter

        manager = MagicMock()
        manager.trace = MagicMock(return_value=None)
        reporter = ErrorReporter()
        with LogContext(user_id="alice@clinic.example", session_id=ALICE):
            reporter._report_via_provider_manager(
                manager, reporter._build_error_data(ValueError("boom"))
            )
        return manager.trace.call_args.kwargs

    def test_the_user_id_is_pseudonymised(self, keyed, monkeypatch):
        assert self._report(monkeypatch)["user_id"].startswith("id#")

    def test_the_session_id_is_pseudonymised(self, keyed, monkeypatch):
        kwargs = self._report(monkeypatch)
        assert ALICE not in str(kwargs["session_id"])
        assert kwargs["session_id"].startswith("id#")

    def test_it_matches_the_log_rendering(self, keyed, monkeypatch):
        """Same pseudonym in Langfuse and in the log, or the two cannot be joined."""
        with LogContext(session_id=ALICE):
            in_log = _context_for_output()["session_id"]
        assert self._report(monkeypatch)["session_id"] == in_log


# ── one id, one pseudonym, on every line ──────────────────────────────────────


class TestOneIdOnePseudonym:
    """The promise log_id() exists to keep: the same person renders the same way
    everywhere, so an operator can follow them through a request.

    It was broken in two shapes, found by reading a live log in which user "tl"
    appeared as three unrelated pseudonyms on consecutive lines:

      containers  log_id(scope) and log_id(identifiers) pseudonymised the
                  repr of a MemoryScope and of a dict -- a pseudonym of
                  "{'user_id': 'tl'}", which matches nothing else.
      slices      log_id(session_id[:8]) pseudonymised an eight-character
                  prefix, which never matches the full id's pseudonym.

    The canary could not see either: it asserts the id is absent, and in both
    cases it was. Absent is necessary, not sufficient.
    """

    def test_a_mapping_pseudonymises_each_value_not_its_repr(self, keyed):
        """So the user inside {'user_id': 'tl'} is the same user as 'tl'."""
        user = str(log_id("alice@clinic.example"))
        rendered = str(log_id({"user_id": "alice@clinic.example"}))
        assert user in rendered, rendered
        assert "alice" not in rendered

    def test_a_mapping_keeps_its_keys(self, keyed):
        """The keys are the structure -- which identifier is which."""
        rendered = str(log_id({"user_id": "a", "agent_id": "b"}))
        assert "user_id" in rendered and "agent_id" in rendered

    def test_a_mapping_is_revealed_whole_when_asked(self, revealed):
        mapping = {"user_id": "alice@clinic.example"}
        assert _filtered(_record("ids=%s", log_id(mapping))) == f"ids={mapping}"

    def test_a_mapping_without_a_secret_withholds_each_value(self, unkeyed):
        rendered = str(log_id({"user_id": "alice@clinic.example"}))
        assert "alice" not in rendered
        assert "user_id" in rendered

    def test_the_memory_search_line_names_the_same_user_as_the_result_line(
        self, keyed, captured_json
    ):
        """The line that exposed this. Drives the real MemoryClient.search."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        from continuum.memory.base import BaseMemoryProvider
        from continuum.memory.client import MemoryClient
        from continuum.memory.config import MemoryConfig

        provider = MagicMock(spec=BaseMemoryProvider)
        provider.search = AsyncMock(return_value=MagicMock(results=[], total_results=0))
        client = MemoryClient(config=MemoryConfig(), provider=provider, auto_initialize=False)
        client._initialized = True

        asyncio.run(client.search("what did I say?", user_id="alice@clinic.example"))

        user = str(log_id("alice@clinic.example"))
        search_lines = [
            line["message"]
            for line in _lines(captured_json)
            if "MEMORY CLIENT SEARCH:" in line["message"]
        ]
        assert search_lines, "the search line was not emitted"
        assert user in search_lines[0], search_lines[0]


class TestNoPseudonymOfSomethingElse:
    """Structural, because the behavioural tests reach a handful of the ninety
    log_id() sites and these two shapes are recognisable without running them."""

    @staticmethod
    def _log_id_args():
        import ast
        from pathlib import Path

        root = Path(__file__).resolve().parents[2] / "src" / "continuum"
        for path in root.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "log_id"
                    and node.args
                ):
                    yield f"{path.name}:{node.lineno}", node.args[0]

    def test_no_slice_is_pseudonymised(self):
        """log_id(x[:8]) is a pseudonym of a prefix, joinable with nothing."""
        import ast

        offenders = [loc for loc, arg in self._log_id_args() if isinstance(arg, ast.Subscript)]
        assert offenders == [], offenders

    def test_no_logger_argument_is_sliced_at_all(self):
        """The broader rule, and the one that would have caught the case the
        narrow one missed: run_finalizer logged original[:8] -- a bare prefix of
        the session id, i.e. on plaintext defaults "u:alice@", the front of an
        email -- with no wrapper at all, so a check on log_id() arguments could
        not see it.

        A slice in a log argument is wrong whatever the value is. On content it
        leaks a prefix and discards the rest; on an identifier it is a partial
        leak and matches the full id on no other line. The docs already say
        "no [:200] slicing"; this makes it true."""
        import ast
        from pathlib import Path

        root = Path(__file__).resolve().parents[2] / "src" / "continuum"
        offenders = []
        for path in root.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                    continue
                if node.func.attr not in ("info", "debug", "warning", "error", "critical"):
                    continue
                for arg in node.args[1:]:
                    if any(
                        isinstance(sub, ast.Subscript) and isinstance(sub.slice, ast.Slice)
                        for sub in ast.walk(arg)
                    ):
                        offenders.append(f"{path.name}:{node.lineno} {ast.unparse(arg)}")
        assert offenders == [], offenders

    def test_no_fallback_literal_is_pseudonymised(self):
        """log_id(x if x else "none") pseudonymises the word "none" when x is
        absent -- a stable id for a user who does not exist. The wrapper belongs
        inside the branch: log_id(x) if x else "none"."""
        import ast

        offenders = [loc for loc, arg in self._log_id_args() if isinstance(arg, ast.IfExp)]
        assert offenders == [], offenders


# ── the two wrappers do not collide ───────────────────────────────────────────


class TestBothWrappers:
    def test_content_and_identity_in_one_line(self, keyed):
        rendered = _filtered(
            _record("session=%s prompt=%s", log_id(ALICE), log_content("PATIENT NOTE"))
        )
        assert ALICE not in rendered
        assert "PATIENT NOTE" not in rendered
        assert "id#" in rendered
        assert "<12 chars>" in rendered
