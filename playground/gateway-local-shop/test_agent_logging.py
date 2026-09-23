"""The demo's own session line must not print the session id.

Lives beside the demo rather than under tests/, like test_memory_list.py: it
asserts something about *this playground's* code, and a bare `pytest` does not
collect it (testpaths = ["tests"]). Run it by path:

    pytest playground/gateway-local-shop/test_agent_logging.py

Found in a live run of `python server.py` + `python web.py`:

    continuum.session.client - Session ready: id#7462a5346f3055bb
    continuum.agent          - ✓ Active Session ID: s_090c2a5a62f2ec374e18155d1efc0da1

Same session, two renderings. The SDK line is pseudonymised; the demo's is the
raw storage key, printed with an f-string -- and playground/ is exempt from ruff
G004, so nothing had flagged it. With SESSION_HASH_IDS=true (hence "s_") the key
is opaque, so nothing personal leaked in that run. But the two lines cannot be
joined, and with hashing off -- the shipped default -- the same line prints
"c:<conversation>:u:<user_id>", the user's id in plaintext.

No servers and no model: chat() is driven over stubs, so this is about the one
log line only.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

USER = "ZZQX-CANARY-7f3a"
# The shipped-default shape: plaintext, derived from the user id.
PLAINTEXT_SESSION = f"c:conv-1:u:{USER}"


@pytest.fixture
def logged(monkeypatch):
    from continuum.config import settings

    monkeypatch.setattr(settings, "log_prompt_content", False)
    monkeypatch.setattr(settings, "session_id_secret", "0123456789abcdef" * 4)

    lines: list[str] = []

    class Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            lines.append(record.getMessage())

    root = logging.getLogger("continuum")
    handler, level = Collector(), root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        yield lines
    finally:
        root.removeHandler(handler)
        root.setLevel(level)


def _shop(monkeypatch, *, session_id: str | None = None, session_error: Exception | None = None):
    """A LocalShopAgent over stubs, whose session setup returns or raises."""
    import agent as agent_module

    # Inert here, and irrelevant to the lines under test.
    monkeypatch.setattr(agent_module, "timing_turn", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(agent_module, "_turn_meta", lambda *a, **k: {})

    shop = agent_module.LocalShopAgent.__new__(agent_module.LocalShopAgent)
    shop._initialized = True
    shop._mcp_server = None
    shop._tool_executor = None
    shop._agent = MagicMock()
    shop.config = MagicMock()

    session_client = MagicMock(is_enabled=True)
    session_client.get_or_create_session = AsyncMock(
        return_value=session_id, side_effect=session_error
    )
    shop._container = MagicMock(session_client=session_client)

    async def _no_events(*args, **kwargs):
        return
        yield  # an async generator that yields nothing

    shop._runner = MagicMock()
    shop._runner._session_service.load_tool_context_state = AsyncMock(return_value=MagicMock())
    shop._runner._session_service.save_tool_context_state = AsyncMock()
    shop._runner.run = AsyncMock(return_value=MagicMock(content="ok"))
    shop._runner.run_stream = _no_events
    return shop


def _run_chat(monkeypatch, session_id: str) -> None:
    shop = _shop(monkeypatch, session_id=session_id)
    asyncio.run(shop.chat("hello", user_id=USER, conversation_id="conv-1"))


def _drive(shop, method: str) -> None:
    """Run chat() or drain chat_stream(), whichever is under test."""
    if method == "chat":
        asyncio.run(shop.chat("hello", user_id=USER, conversation_id="conv-1"))
        return

    async def drain():
        async for _ in shop.chat_stream("hello", user_id=USER, conversation_id="conv-1"):
            pass

    asyncio.run(drain())


def _session_line(lines: list[str]) -> str:
    found = [line for line in lines if "Active Session ID" in line]
    assert found, f"the session line was not emitted: {lines}"
    return found[0]


class TestActiveSessionLine:
    def test_a_plaintext_session_id_does_not_reach_the_log(self, logged, monkeypatch):
        """The shipped default. The id contains the user's id verbatim."""
        _run_chat(monkeypatch, PLAINTEXT_SESSION)
        assert USER not in _session_line(logged)

    def test_it_matches_the_sdk_s_rendering_of_the_same_session(self, logged, monkeypatch):
        """The point of a pseudonym: this line and the SDK's
        "Session ready: id#..." line name the session the same way, so an
        operator can follow one request across both."""
        from continuum.logging import log_id

        _run_chat(monkeypatch, PLAINTEXT_SESSION)
        assert str(log_id(PLAINTEXT_SESSION)) in _session_line(logged)

    def test_a_hashed_session_id_is_pseudonymised_too(self, logged, monkeypatch):
        """With SESSION_HASH_IDS=true the key is already opaque, so this is about
        joinability rather than privacy: the raw s_ key never appears on any SDK
        line, so printing it here gave a value nothing else could be matched to."""
        from continuum.logging import log_id

        hashed = "s_090c2a5a62f2ec374e18155d1efc0da1"
        _run_chat(monkeypatch, hashed)
        line = _session_line(logged)
        assert hashed not in line
        assert str(log_id(hashed)) in line


class TestSessionFailureLines:
    """The same try block's error branches, in both chat() and chat_stream().

    They printed the user id itself -- not a session id derived from it, the
    raw value -- two lines below the Active Session ID line that was fixed to
    hide it. They only fire when session setup fails (Redis down, a weak
    SESSION_ID_SECRET), so a normal run never shows them; that is how they
    stayed out of the screenshot that found the first one. And WARNING and
    ERROR are the levels that get forwarded, kept longest, and pasted into
    tickets.

    Pseudonymised rather than removed: "this user keeps failing session init"
    stays readable, because the pseudonym is stable and matches the one every
    SDK line uses for the same user.
    """

    CASES = [
        ("chat", "insecure"),
        ("chat", "failure"),
        ("chat_stream", "insecure"),
        ("chat_stream", "failure"),
    ]

    @staticmethod
    def _error(kind: str) -> Exception:
        from continuum.exceptions import InsecureConfigurationError

        if kind == "insecure":
            return InsecureConfigurationError("SESSION_ID_SECRET is too short")
        return ConnectionError("redis unreachable")

    @staticmethod
    def _failure_line(lines: list[str], kind: str) -> str:
        marker = "Insecure session config" if kind == "insecure" else "Session init failed"
        found = [line for line in lines if marker in line]
        assert found, f"the {kind} line was not emitted: {lines}"
        return found[0]

    @pytest.mark.parametrize("method,kind", CASES)
    def test_the_user_id_does_not_reach_the_log(self, logged, monkeypatch, method, kind):
        _drive(_shop(monkeypatch, session_error=self._error(kind)), method)
        assert USER not in self._failure_line(logged, kind)

    @pytest.mark.parametrize("method,kind", CASES)
    def test_it_names_the_user_as_every_sdk_line_does(self, logged, monkeypatch, method, kind):
        from continuum.logging import log_id

        _drive(_shop(monkeypatch, session_error=self._error(kind)), method)
        assert str(log_id(USER)) in self._failure_line(logged, kind)

    @pytest.mark.parametrize("method,kind", CASES)
    def test_the_reason_survives(self, logged, monkeypatch, method, kind):
        """The exception is the diagnostic. It must still be on the line."""
        error = self._error(kind)
        _drive(_shop(monkeypatch, session_error=error), method)
        assert str(error) in self._failure_line(logged, kind)
