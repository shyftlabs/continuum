"""Two-file tool-trust storage (security finding F3, phase 1).

The pin file used to serve two roles with opposite mutability requirements:

  * the drift tripwire needs a *mutable* baseline ("what I saw last time"),
    otherwise it re-warns forever after one legitimate update;
  * the pinning gate needs an *immutable* one ("what a human approved"),
    otherwise it is not an approval.

Sharing one file meant the tripwire rewrote what the gate reads, so an
attacker's catalogue was silently promoted to "approved" and the gate disarmed
itself on the next restart -- warning on run 1, silent on run 2.

These tests pin the split: an approved file only a human command writes, and a
derived last-seen file only the runtime writes.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.types import Tool

from continuum.tools.exceptions import MCPConnectionError, MCPError, MCPServerUnreviewedError
from continuum.tools.mcp import MCPServerStreamableHttp
from continuum.tools.pinning import PIN_FORMAT_VERSION, load_pins, save_pins, snapshot_tool_digests
from continuum.tools.types import ToolTrustConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool(name: str, description: str, schema: dict | None = None) -> Tool:
    return Tool(
        name=name,
        description=description,
        inputSchema=schema or {"type": "object", "properties": {}},
    )


def _make_server(pin_path: str | Path | None = None, **kwargs) -> MCPServerStreamableHttp:
    trust = kwargs.pop("trust_config", None)
    if trust is None and pin_path is not None:
        # on_unreviewed="allow" so these tests exercise *storage* in isolation.
        # The default blocks a server with no approved catalogue, which is
        # correct (see test_tool_trust_enforcement.py) but would stop several
        # of these before they reach the file they are about.
        trust = ToolTrustConfig(pin_path=pin_path, on_unreviewed="allow")
    return MCPServerStreamableHttp(
        params={"url": "http://localhost:8888/mcp"},
        cache_tools_list=False,
        name="srv",
        trust_config=trust,
        **kwargs,
    )


def _attach_session(server, tools: list[Tool]) -> MagicMock:
    session = AsyncMock()
    result = MagicMock()
    result.tools = tools
    session.list_tools = AsyncMock(return_value=result)
    server.session = session
    return session


@contextmanager
def _captured_logs(name: str = "continuum.tools"):
    """Collect records from a continuum logger.

    pytest's caplog cannot see them: the "continuum" parent sets
    propagate=False and owns its handler, so nothing reaches the root logger.
    """
    records: list[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Collector()
    loggers = [logging.getLogger(f"{name}.mcp"), logging.getLogger(f"{name}.pinning")]
    for logger in loggers:
        logger.addHandler(handler)
    try:
        yield records
    finally:
        for logger in loggers:
            logger.removeHandler(handler)


def _warnings(records: list[logging.LogRecord]) -> list[str]:
    return [r.getMessage() for r in records if r.levelno >= logging.WARNING]


# ---------------------------------------------------------------------------
# A. Pin file format
# ---------------------------------------------------------------------------


class TestPinFileFormat:
    """The file is a lockfile: versioned, deterministic, and human-readable.

    Readable matters as much as versioned -- a hash records *that* something
    changed and can never record *what*, so a reviewer handed only digests has
    nothing to review.
    """

    def test_save_writes_a_version_and_a_servers_key(self, tmp_path):
        path = tmp_path / "tool-pins.json"
        save_pins(path, {"srv": snapshot_tool_digests("srv", [_tool("a", "does a")])})

        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["version"] == PIN_FORMAT_VERSION
        assert "srv" in raw["servers"]

    def test_round_trips(self, tmp_path):
        path = tmp_path / "tool-pins.json"
        servers = {"srv": snapshot_tool_digests("srv", [_tool("a", "does a")])}
        save_pins(path, servers)

        assert load_pins(path) == servers

    def test_entry_carries_description_and_schema_text_not_only_digests(self, tmp_path):
        path = tmp_path / "tool-pins.json"
        schema = {"type": "object", "properties": {"q": {"type": "string"}}}
        save_pins(path, {"srv": snapshot_tool_digests("srv", [_tool("a", "does a", schema)])})

        entry = load_pins(path)["srv"]["a"]
        assert entry["description"] == "does a"
        assert entry["inputSchema"] == schema
        assert entry["raw"] and entry["effective"]

    def test_missing_file_is_empty_not_an_error(self, tmp_path):
        assert load_pins(tmp_path / "nope.json") == {}

    def test_corrupt_json_degrades_with_a_warning(self, tmp_path):
        path = tmp_path / "tool-pins.json"
        path.write_text("{not json", encoding="utf-8")

        with _captured_logs() as records:
            assert load_pins(path) == {}
        assert any("tool-pins.json" in w for w in _warnings(records))

    def test_unknown_future_version_is_refused_rather_than_misparsed(self, tmp_path):
        """A newer writer may mean anything by these keys.

        Guessing is how the dead single-digest compat branch was born: a format
        with no version field, read by code that had to infer the shape.
        """
        path = tmp_path / "tool-pins.json"
        path.write_text(
            json.dumps({"version": PIN_FORMAT_VERSION + 1, "servers": {"srv": {}}}),
            encoding="utf-8",
        )

        with _captured_logs() as records:
            assert load_pins(path) == {}
        assert any("version" in w for w in _warnings(records))

    def test_saving_one_server_preserves_the_others(self, tmp_path):
        path = tmp_path / "tool-pins.json"
        save_pins(path, {"a": snapshot_tool_digests("a", [_tool("t", "one")])})

        existing = load_pins(path)
        existing["b"] = snapshot_tool_digests("b", [_tool("t", "two")])
        save_pins(path, existing)

        assert set(load_pins(path)) == {"a", "b"}

    def test_serialization_is_deterministic(self, tmp_path):
        """Key order must not depend on insertion order.

        The file is meant to be diffed; a reordering diff is a diff nobody reads.
        """
        first = tmp_path / "one.json"
        second = tmp_path / "two.json"
        tools = [_tool("b", "bee"), _tool("a", "ay")]
        save_pins(first, {"srv": snapshot_tool_digests("srv", tools)})
        save_pins(second, {"srv": snapshot_tool_digests("srv", list(reversed(tools)))})

        assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# B. Two files, one writer each
# ---------------------------------------------------------------------------


class TestRuntimeNeverWritesTheApprovedFile:
    """The invariant the whole split exists to enforce.

    If the runtime can write the approved file, then a poisoned catalogue
    observed on run 1 becomes the approval read on run 2, and the gate passes
    it while reporting nothing.
    """

    async def test_approved_file_is_untouched_by_a_fetch(self, tmp_path):
        path = tmp_path / "tool-pins.json"
        save_pins(path, {"srv": snapshot_tool_digests("srv", [_tool("a", "original")])})
        before = path.read_text(encoding="utf-8")

        server = _make_server(pin_path=path)
        _attach_session(server, [_tool("a", "POISONED")])
        await server.list_tools()

        assert path.read_text(encoding="utf-8") == before

    async def test_approved_file_is_not_created_when_absent(self, tmp_path):
        path = tmp_path / "tool-pins.json"
        server = _make_server(pin_path=path)
        _attach_session(server, [_tool("a", "hello")])
        await server.list_tools()

        assert not path.exists()

    async def test_runtime_records_into_the_derived_last_seen_file(self, tmp_path):
        path = tmp_path / "tool-pins.json"
        server = _make_server(pin_path=path)
        _attach_session(server, [_tool("a", "hello")])
        await server.list_tools()

        assert server.trust_config.last_seen_path.exists()

    async def test_last_seen_path_is_a_sibling_derived_from_the_pin_path(self, tmp_path):
        cfg = ToolTrustConfig(pin_path=tmp_path / "tool-pins.json")

        assert cfg.last_seen_path.parent == tmp_path
        assert cfg.last_seen_path.name.startswith(".")
        assert cfg.last_seen_path != Path(cfg.pin_path)

    async def test_no_pin_path_writes_nothing_at_all(self, monkeypatch):
        """Memory-only is the default; a library must not create files unasked.

        Spies on save_pins rather than checking a directory: with pin_path=None
        there is no directory the server knows about, so "nothing appeared in
        tmp_path" would pass no matter what the code did.
        """
        calls: list[tuple] = []
        monkeypatch.setattr(
            "continuum.tools.pinning.save_pins",
            lambda path, servers: calls.append((path, servers)),
        )

        server = _make_server(pin_path=None)
        _attach_session(server, [_tool("a", "hello")])
        await server.list_tools()

        assert calls == []


class TestDegradesRatherThanFails:
    """Trust bookkeeping is best-effort; enforcement is not.

    A read-only production filesystem must lose the drift *diagnostic* without
    losing the *gate*, so an unwritable last-seen file is a log line.
    """

    async def test_unwritable_last_seen_warns_and_still_lists_tools(self, tmp_path):
        pin = tmp_path / "sub" / "tool-pins.json"
        pin.parent.mkdir()
        pin.parent.chmod(0o500)  # readable, not writable
        try:
            server = _make_server(pin_path=pin)
            _attach_session(server, [_tool("a", "hello")])

            with _captured_logs() as records:
                tools = await server.list_tools()

            assert [t.name for t in tools] == ["a"]
            assert any(
                server.trust_config.last_seen_path.name in w for w in _warnings(records)
            ), _warnings(records)
        finally:
            pin.parent.chmod(0o700)

    async def test_unreadable_approved_file_does_not_break_listing(self, tmp_path):
        pin = tmp_path / "tool-pins.json"
        pin.write_text("{ corrupt", encoding="utf-8")

        server = _make_server(pin_path=pin)
        _attach_session(server, [_tool("a", "hello")])

        with _captured_logs():
            tools = await server.list_tools()
        assert [t.name for t in tools] == ["a"]


class TestLastSeenFallsBackToApproved:
    """A fresh machine has an approved catalogue but no last-seen file.

    Treating that as "no baseline" would make the first run after every deploy
    silent about drift, which is the run most likely to meet a changed server.
    """

    async def test_first_run_on_a_fresh_machine_still_reports_drift(self, tmp_path):
        pin = tmp_path / "tool-pins.json"
        save_pins(pin, {"srv": snapshot_tool_digests("srv", [_tool("a", "original")])})

        server = _make_server(pin_path=pin)
        _attach_session(server, [_tool("a", "CHANGED")])

        with _captured_logs() as records:
            await server.list_tools()

        assert any("'a'" in w or "['a']" in w for w in _warnings(records))

    async def test_matching_catalogue_on_a_fresh_machine_is_silent(self, tmp_path):
        pin = tmp_path / "tool-pins.json"
        save_pins(pin, {"srv": snapshot_tool_digests("srv", [_tool("a", "original")])})

        server = _make_server(pin_path=pin)
        _attach_session(server, [_tool("a", "original")])

        with _captured_logs() as records:
            await server.list_tools()

        assert _warnings(records) == []


# ---------------------------------------------------------------------------
# C. ToolTrustConfig
# ---------------------------------------------------------------------------


class TestToolTrustConfigDefaults:
    """Posture is env-configurable, like every other operational setting.

    An application wants `warn` in development and `block` in production
    without a code change; hard-coding the literals in the dataclass would make
    that a redeploy.
    """

    def test_defaults_are_block_on_unreviewed_and_warn_on_drift(self):
        cfg = ToolTrustConfig()

        assert cfg.on_unreviewed == "block"
        assert cfg.on_drift == "warn"

    def test_defaults_come_from_settings(self, monkeypatch):
        from continuum.config import get_settings

        monkeypatch.setenv("MCP_ON_UNREVIEWED", "allow")
        monkeypatch.setenv("MCP_ON_DRIFT", "block")
        get_settings.cache_clear()
        try:
            assert ToolTrustConfig().on_unreviewed == "allow"
            assert ToolTrustConfig().on_drift == "block"
        finally:
            get_settings.cache_clear()

    def test_explicit_arguments_win_over_settings(self, monkeypatch):
        from continuum.config import get_settings

        monkeypatch.setenv("MCP_ON_UNREVIEWED", "allow")
        get_settings.cache_clear()
        try:
            assert ToolTrustConfig(on_unreviewed="block").on_unreviewed == "block"
        finally:
            get_settings.cache_clear()

    def test_no_pin_path_means_no_last_seen_path(self):
        assert ToolTrustConfig().last_seen_path is None


# ---------------------------------------------------------------------------
# D. MCPServerUnreviewedError
# ---------------------------------------------------------------------------


class TestUnreviewedError:
    def test_is_an_mcp_error(self):
        err = MCPServerUnreviewedError("nope", server_name="clinic")

        assert isinstance(err, MCPError)
        assert err.context["server_name"] == "clinic"

    async def test_connect_does_not_rewrap_it_as_a_connection_error(self, monkeypatch):
        """`validate_on_connect` calls list_tools() inside connect()'s catch-all.

        Without a carve-out the message -- which names the command that fixes
        the problem -- ends up buried in `original_error`, and the reader goes
        debugging the network instead of running `mcp inspect`.
        """
        server = _make_server(validate_on_connect=True)

        async def _boom(*args, **kwargs):
            raise MCPServerUnreviewedError("needs review", server_name="srv")

        monkeypatch.setattr(server, "list_tools", _boom)
        monkeypatch.setattr(server, "create_streams", lambda: _fake_streams())
        monkeypatch.setattr(
            "continuum.tools.mcp.ClientSession", lambda *a, **k: _FakeSession()
        )

        with pytest.raises(MCPServerUnreviewedError):
            await server.connect()

    async def test_ordinary_failures_are_still_wrapped(self, monkeypatch):
        server = _make_server(validate_on_connect=True)

        async def _boom(*args, **kwargs):
            raise RuntimeError("socket exploded")

        monkeypatch.setattr(server, "list_tools", _boom)
        monkeypatch.setattr(server, "create_streams", lambda: _fake_streams())
        monkeypatch.setattr(
            "continuum.tools.mcp.ClientSession", lambda *a, **k: _FakeSession()
        )

        with pytest.raises(MCPConnectionError):
            await server.connect()


# --- minimal async-context doubles for connect() ---------------------------


class _FakeSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def initialize(self):
        return MagicMock()


def _fake_streams():
    class _Ctx:
        async def __aenter__(self):
            return (MagicMock(), MagicMock(), None)

        async def __aexit__(self, *exc):
            return False

    return _Ctx()
