"""Tests for SessionClient.add_message memory-write routing (sync vs background).

Verifies:
  - default 'sync' mode awaits the mem0 write before returning (no behavior change),
  - 'background' mode returns before the mem0 write completes (the latency win),
  - the short-term Redis write is always synchronous,
  - background mode falls back to synchronous when no registry is available.

All collaborators are mocked — no Redis, no mem0/LLM, no vector store.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from continuum.core.background_tasks import BackgroundTaskRegistry
from continuum.llm.types import ChatMessage
from continuum.session.client import SessionClient
from continuum.session.config import SessionConfig
from continuum.session.principal import bind_principal
from continuum.session.types import SessionMetadata


@pytest.fixture(autouse=True)
def _identify_as_the_session_owner():
    """Every test here drives a session owned by ``user-1`` (see ``_metadata``).

    Session ownership is enforced by default, so a caller must say who it is —
    exactly what a real application does at its auth boundary. These tests are
    about memory-write routing, not ownership, so they identify once here
    rather than repeating it in twenty places.
    """
    with bind_principal("user-1"):
        yield


def _metadata(session_id="sess-1234abcd"):
    now = datetime.now(UTC)
    return SessionMetadata(
        session_id=session_id,
        user_id="user-1",
        agent_id="agent-1",
        conversation_id="conv-1",
        created_at=now,
        last_accessed_at=now,
    )


def _make_client(*, mode, with_registry=True, add_mock=None):
    """Build a SessionClient with mocked provider + memory client."""
    mem = MagicMock()
    mem.is_enabled = True
    mem.add = add_mock or AsyncMock(return_value=MagicMock(results=[]))
    mem.delete = AsyncMock()

    provider = MagicMock()
    provider.add_message = AsyncMock()
    provider.get_session_metadata = AsyncMock(return_value=_metadata())

    registry = BackgroundTaskRegistry(name="test") if with_registry else None

    client = SessionClient(
        session_config=SessionConfig(enabled=True, memory_write_mode=mode),
        memory_client=mem,
        provider=provider,
        auto_initialize=False,
        background_tasks=registry,
    )
    return client, mem, provider, registry


def _msg():
    return ChatMessage(role="user", content="My name is Tom")


def _gated_add(gate):
    """An AsyncMock whose add() blocks until `gate` is set, then returns a result."""

    async def _add(*args, **kwargs):
        await gate.wait()
        return MagicMock(results=[])

    return AsyncMock(side_effect=_add)


class TestSyncMode:
    async def test_shipped_default_is_background(self):
        # The shipped default is 'background' (writes off the response path).
        # Isolate from any ambient SESSION_MEMORY_WRITE_MODE in the dev .env/shell
        # by pinning the setting the field reads from.
        with patch("continuum.session.config.settings") as mock_settings:
            mock_settings.session_memory_write_mode = "background"
            # Patching the whole settings object turns every unpinned field into
            # a MagicMock, and pydantic does not validate default_factory values —
            # so the session-id-secret validator would otherwise be handed one.
            mock_settings.session_hash_ids = False
            mock_settings.session_id_secret = None
            assert SessionConfig().memory_write_mode == "background"

    async def test_sync_awaits_memory_add_before_return(self):
        client, mem, provider, _ = _make_client(mode="sync")
        await client.add_message("sess-1234abcd", _msg())
        provider.add_message.assert_awaited_once()
        mem.add.assert_awaited_once()  # completed inline, before return

    async def test_sync_blocks_until_memory_write_finishes(self):
        gate = asyncio.Event()
        slow_add = _gated_add(gate)
        client, mem, _, _ = _make_client(mode="sync", add_mock=slow_add)

        call = asyncio.ensure_future(client.add_message("sess-1234abcd", _msg()))
        # Should NOT complete while the memory write is blocked.
        done, _pending = await asyncio.wait({call}, timeout=0.1)
        assert call not in done
        # Release and let it finish.
        gate.set()
        await call


class TestBackgroundMode:
    async def test_background_returns_before_memory_add_completes(self):
        gate = asyncio.Event()
        slow_add = _gated_add(gate)
        client, mem, provider, registry = _make_client(mode="background", add_mock=slow_add)

        await client.add_message("sess-1234abcd", _msg())  # returns immediately

        # Redis write happened synchronously; mem0 write was scheduled, not awaited.
        provider.add_message.assert_awaited_once()
        assert mem.add.await_count == 0  # not yet completed
        assert len(registry) == 1  # in flight

        # Let it finish and confirm it eventually ran.
        gate.set()
        await registry.drain(timeout=1.0)
        mem.add.assert_awaited_once()
        assert len(registry) == 0

    async def test_redis_write_always_synchronous_in_background_mode(self):
        # The conversation-history write must not be backgrounded.
        client, mem, provider, registry = _make_client(mode="background")
        await client.add_message("sess-1234abcd", _msg())
        provider.add_message.assert_awaited_once()
        await registry.drain(timeout=1.0)

    async def test_falls_back_to_sync_when_no_registry(self):
        # background mode + no registry (and container unavailable) → inline write.
        client, mem, _, _ = _make_client(mode="background", with_registry=False)
        with patch(
            "continuum.core.container.get_container",
            side_effect=RuntimeError("no container"),
        ):
            await client.add_message("sess-1234abcd", _msg())
        mem.add.assert_awaited_once()  # ran inline


class TestTemporalDowngrade:
    async def test_temporal_activity_forces_sync_even_in_background_mode(self):
        # Inside a Temporal activity, background mode is downgraded to sync so the
        # write completes within the durable/retriable activity boundary.
        client, mem, provider, registry = _make_client(mode="background")
        with patch("continuum.session.client._in_temporal_activity", return_value=True):
            await client.add_message("sess-1234abcd", _msg())
        mem.add.assert_awaited_once()  # ran inline (sync)
        assert len(registry) == 0  # nothing was scheduled in the background

    async def test_non_temporal_uses_background(self):
        # Outside Temporal, background mode schedules the write off the path.
        gate = asyncio.Event()
        client, mem, provider, registry = _make_client(mode="background", add_mock=_gated_add(gate))
        with patch("continuum.session.client._in_temporal_activity", return_value=False):
            await client.add_message("sess-1234abcd", _msg())
        assert mem.add.await_count == 0  # not awaited inline
        assert len(registry) == 1  # scheduled in background
        gate.set()
        await registry.drain(timeout=1.0)
        mem.add.assert_awaited_once()


class TestStoreInMemoryIsBestEffort:
    async def test_memory_failure_does_not_raise(self):
        failing_add = AsyncMock(side_effect=RuntimeError("mem0 down"))
        client, _, provider, _ = _make_client(mode="sync", add_mock=failing_add)
        # Must not raise — memory storage is best-effort.
        await client.add_message("sess-1234abcd", _msg())
        provider.add_message.assert_awaited_once()


class TestPolicyDenialIsQuiet:
    """A data-label policy blocking the long-term write is EXPECTED — it must be
    logged quietly (INFO, no traceback) and never escalated to error reporting."""

    async def test_memory_access_denied_logged_quietly_not_reported(self):
        from continuum.agent.exceptions import MemoryAccessDeniedError
        from continuum.session import client as sc_mod

        denied = AsyncMock(
            side_effect=MemoryAccessDeniedError(
                operation="write", scope="agent", policy_name="phi-never-persisted"
            )
        )
        client, _, provider, _ = _make_client(mode="sync", add_mock=denied)

        with (
            patch("continuum.session.client.report_error") as rep,
            patch.object(sc_mod.logger, "error") as err,
            patch.object(sc_mod.logger, "info") as info,
        ):
            await client.add_message("sess-1234abcd", _msg())  # must NOT raise

        provider.add_message.assert_awaited_once()  # short-term write still happened
        rep.assert_not_called()  # not escalated to error reporting
        err.assert_not_called()  # no ERROR-level log / traceback
        # the expected denial was logged quietly at INFO, with the policy name
        matched = [c for c in info.call_args_list if "blocked by policy" in c.args[0]]
        assert matched and matched[0].args[1] == "phi-never-persisted"


# ---------------------------------------------------------------------------
# pre_store_filter must fail CLOSED (security finding F6)
#
# The hook is named for a gate before the write, but it runs after one: mem0
# extracts and stores in a single call, so the filter can only delete what was
# already persisted. That ordering is not fixable here -- neither mem0 1.0.11 nor
# 2.0.19 exposes an extract-without-store path, and both fuse extraction and
# storage inside one private method with no hook between them.
#
# What is fixable is that every failure in the undo path failed OPEN, quietly:
# a filter that raised kept everything, a fact with no id was skipped in silence,
# and a delete that failed was logged at warning and then reported to on_stored
# as though it had been removed. The exposure window from the ordering is ~280ms
# (measured against the running Milvus); the exposure from these is permanent.
#
# So: a filter that cannot answer rejects everything, a delete that cannot be
# confirmed keeps the fact in the stored list rather than pretending otherwise,
# and neither is whispered at warning level.
# ---------------------------------------------------------------------------


def _fact(text, fact_id):
    return {"memory": text, "id": fact_id}


def _add_returning(*facts):
    return AsyncMock(return_value=MagicMock(results=list(facts)))


class TestFilterFailsClosed:
    async def test_rejected_fact_is_deleted(self):
        client, mem, _, _ = _make_client(
            mode="sync", add_mock=_add_returning(_fact("keep me", "id-1"), _fact("drop me", "id-2"))
        )
        await client.add_message(
            "sess-1234abcd",
            _msg(),
            pre_store_filter=lambda texts: [t for t in texts if t == "keep me"],
        )
        deleted = [c.args[0] for c in mem.delete.await_args_list]
        assert deleted == ["id-2"]

    async def test_filter_that_raises_deletes_everything(self):
        """A filter exists to exclude. If it cannot run, nothing is known about
        what was just written, so none of it may stay."""

        def boom(_texts):
            raise RuntimeError("scanner unavailable")

        client, mem, _, _ = _make_client(
            mode="sync", add_mock=_add_returning(_fact("a", "id-1"), _fact("b", "id-2"))
        )
        await client.add_message("sess-1234abcd", _msg(), pre_store_filter=boom)

        assert sorted(c.args[0] for c in mem.delete.await_args_list) == ["id-1", "id-2"]

    async def test_filter_that_raises_is_logged_at_error(self):
        from continuum.session import client as sc_mod

        def boom(_texts):
            raise RuntimeError("scanner unavailable")

        client, _, _, _ = _make_client(mode="sync", add_mock=_add_returning(_fact("a", "id-1")))
        with patch.object(sc_mod.logger, "error") as err:
            await client.add_message("sess-1234abcd", _msg(), pre_store_filter=boom)

        assert err.called, "a filter that cannot run is not a warning-level event"

    async def test_filter_that_raises_reports_nothing_as_stored(self):
        stored: list[list[str]] = []

        def boom(_texts):
            raise RuntimeError("scanner unavailable")

        client, _, _, _ = _make_client(mode="sync", add_mock=_add_returning(_fact("a", "id-1")))
        await client.add_message(
            "sess-1234abcd", _msg(), pre_store_filter=boom, on_stored=stored.append
        )
        assert stored == [] or stored == [[]]


class TestUnconfirmedDeletesAreNotReportedAsRemoved:
    async def test_fact_whose_delete_failed_stays_in_on_stored(self):
        """on_stored is the developer's record of what is in the store. Listing a
        fact as removed when the delete failed makes that record a lie, and the
        lie is the reason nobody goes looking for the row."""
        stored: list[list[str]] = []
        client, mem, _, _ = _make_client(
            mode="sync", add_mock=_add_returning(_fact("keep", "id-1"), _fact("drop", "id-2"))
        )
        mem.delete = AsyncMock(side_effect=RuntimeError("milvus refused"))

        await client.add_message(
            "sess-1234abcd",
            _msg(),
            pre_store_filter=lambda texts: [t for t in texts if t == "keep"],
            on_stored=stored.append,
        )

        assert stored, "on_stored should still fire"
        assert "drop" in stored[0], "the undeleted fact is still in the store, so say so"

    async def test_failed_delete_is_logged_at_error(self):
        from continuum.session import client as sc_mod

        client, mem, _, _ = _make_client(
            mode="sync", add_mock=_add_returning(_fact("drop", "id-2"))
        )
        mem.delete = AsyncMock(side_effect=RuntimeError("milvus refused"))

        with patch.object(sc_mod.logger, "error") as err:
            await client.add_message("sess-1234abcd", _msg(), pre_store_filter=lambda _t: [])

        assert err.called, "a fact the filter rejected but that is still stored is an error"

    async def test_fact_with_no_id_cannot_be_deleted_and_is_reported(self):
        """mem0 does not always return an id. Silently skipping the delete left a
        rejected fact in the store with nothing said about it."""
        from continuum.session import client as sc_mod

        stored: list[list[str]] = []
        client, mem, _, _ = _make_client(mode="sync", add_mock=_add_returning(_fact("drop", None)))

        with patch.object(sc_mod.logger, "error") as err:
            await client.add_message(
                "sess-1234abcd", _msg(), pre_store_filter=lambda _t: [], on_stored=stored.append
            )

        assert not mem.delete.await_args_list, "nothing to delete with"
        assert err.called, "an undeletable rejected fact must be reported"
        assert stored and "drop" in stored[0]

    async def test_successful_delete_removes_the_fact_from_on_stored(self):
        stored: list[list[str]] = []
        client, _, _, _ = _make_client(
            mode="sync", add_mock=_add_returning(_fact("keep", "id-1"), _fact("drop", "id-2"))
        )
        await client.add_message(
            "sess-1234abcd",
            _msg(),
            pre_store_filter=lambda texts: [t for t in texts if t == "keep"],
            on_stored=stored.append,
        )
        assert stored == [["keep"]]


class TestRejectedContentIsNotLogged:
    async def test_rejected_fact_text_never_reaches_the_log(self):
        """The filter's whole purpose is to keep this text out of persistent
        stores. Echoing it into the application log defeats that, and logs are
        usually the less guarded of the two."""
        from continuum.session import client as sc_mod

        secret = "SSN 123-45-6789"
        client, _, _, _ = _make_client(mode="sync", add_mock=_add_returning(_fact(secret, "id-1")))

        with patch.object(sc_mod, "logger", MagicMock()) as log:
            await client.add_message("sess-1234abcd", _msg(), pre_store_filter=lambda _t: [])

        emitted = " | ".join(str(c) for m in log.method_calls for c in [m])
        assert secret not in emitted, f"rejected content leaked into logs: {emitted}"


class TestFilterHappyPathUnchanged:
    async def test_filter_keeping_everything_deletes_nothing(self):
        stored: list[list[str]] = []
        client, mem, _, _ = _make_client(
            mode="sync", add_mock=_add_returning(_fact("a", "id-1"), _fact("b", "id-2"))
        )
        await client.add_message(
            "sess-1234abcd",
            _msg(),
            pre_store_filter=lambda texts: list(texts),
            on_stored=stored.append,
        )
        assert not mem.delete.await_args_list
        assert stored == [["a", "b"]]

    async def test_no_filter_configured_changes_nothing(self):
        stored: list[list[str]] = []
        client, mem, _, _ = _make_client(mode="sync", add_mock=_add_returning(_fact("a", "id-1")))
        await client.add_message("sess-1234abcd", _msg(), on_stored=stored.append)
        assert not mem.delete.await_args_list
        assert stored == [["a"]]


class TestFalsyDeleteCountsAsFailure:
    """``MemoryClient.delete`` reports failure two ways.

    Mem0Provider catches its own exceptions and returns False, so a delete can
    fail without raising. Checking only for an exception meant a rejected fact
    was pruned from `stored_pairs` -- reported to on_stored as removed -- while
    the row stayed in the vector store. That is the precise bug this whole path
    exists to prevent, reintroduced one layer up by watching the wrong signal.

    Seen live: mem0 raised "list index out of range" deleting a row written
    milliseconds earlier, the provider logged it and returned False, and the
    fact the filter had rejected was still searchable ten seconds later.
    """

    async def test_falsy_delete_is_reported_as_still_stored(self):
        from continuum.session import client as sc_mod

        client, mem, _, _ = _make_client(
            mode="sync", add_mock=_add_returning(_fact("drop", "id-2"))
        )
        mem.delete = AsyncMock(return_value=False)  # failed, but did not raise

        with patch.object(sc_mod.logger, "error") as err:
            await client.add_message("sess-1234abcd", _msg(), pre_store_filter=lambda _t: [])

        assert err.called, "a delete that returned False must be reported"

    async def test_falsy_delete_keeps_the_fact_in_on_stored(self):
        stored: list[list[str]] = []
        client, mem, _, _ = _make_client(
            mode="sync", add_mock=_add_returning(_fact("keep", "id-1"), _fact("drop", "id-2"))
        )
        mem.delete = AsyncMock(return_value=False)

        await client.add_message(
            "sess-1234abcd",
            _msg(),
            pre_store_filter=lambda texts: [t for t in texts if t == "keep"],
            on_stored=stored.append,
        )

        assert stored, "on_stored should still fire"
        assert "drop" in stored[0], "the row is still in the store, so say so"

    async def test_truthy_delete_still_counts_as_removed(self):
        stored: list[list[str]] = []
        client, mem, _, _ = _make_client(
            mode="sync", add_mock=_add_returning(_fact("keep", "id-1"), _fact("drop", "id-2"))
        )
        mem.delete = AsyncMock(return_value=True)

        await client.add_message(
            "sess-1234abcd",
            _msg(),
            pre_store_filter=lambda texts: [t for t in texts if t == "keep"],
            on_stored=stored.append,
        )

        assert stored == [["keep"]]

    async def test_none_return_is_not_treated_as_failure(self):
        """A provider that returns nothing at all is the common Python shape for
        "did it"; only an explicitly falsy non-None result means failure."""
        stored: list[list[str]] = []
        client, mem, _, _ = _make_client(
            mode="sync", add_mock=_add_returning(_fact("drop", "id-2"))
        )
        mem.delete = AsyncMock(return_value=None)

        await client.add_message(
            "sess-1234abcd", _msg(), pre_store_filter=lambda _t: [], on_stored=stored.append
        )

        assert stored == [] or stored == [[]]
