"""
Tests for run-state persistence defaulting to OFF via the PERSIST_RUN_STATE flag.

Background: run-state persistence (RunnerConfig.persist_state) writes RunState to
Redis on every run, for a future pause/resume feature that is not yet implemented
(nothing reads the data back). It used to default to True with no env control, so
every integrator had to disable it by hand. It now defaults to the PERSIST_RUN_STATE
env flag, which is off unless explicitly enabled.

Three layers are covered:
  1. Config: the Settings default is off; the env flag flips it; RunnerConfig's
     default tracks settings; an explicit per-runner override still wins.
  2. Behavior: with persist_state off, ContextService never calls the state
     manager's save() (no Redis write / connection attempt); with it on, it does.
  3. Ticket regression: with the default (disabled) config, the run path never
     reaches for the global state manager (so no Redis connection is attempted)
     and emits no "Failed to connect to Redis for state persistence" warning —
     the exact symptom the ticket reported.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from continuum.agent.config import RunnerConfig
from continuum.agent.services.context_service import ContextService

# ---------------------------------------------------------------------------
# 1. Config-level behavior
# ---------------------------------------------------------------------------


class TestPersistStateConfig:
    def test_settings_default_is_off(self, monkeypatch):
        # The Settings *default* (no env override) is False.
        from continuum.config import Settings

        monkeypatch.delenv("PERSIST_RUN_STATE", raising=False)
        assert Settings().persist_run_state is False

    def test_env_flag_enables(self, monkeypatch):
        # Setting PERSIST_RUN_STATE=true flips the setting on.
        from continuum.config import Settings

        monkeypatch.setenv("PERSIST_RUN_STATE", "true")
        assert Settings().persist_run_state is True

    def test_runnerconfig_default_tracks_settings(self):
        # Env-independent: the field must default from global settings, not a
        # hard-coded literal.
        from continuum.config import settings

        assert RunnerConfig().persist_state == settings.persist_run_state

    def test_explicit_override_wins(self):
        # A caller passing persist_state explicitly always wins over the default.
        assert RunnerConfig(persist_state=True).persist_state is True
        assert RunnerConfig(persist_state=False).persist_state is False


# ---------------------------------------------------------------------------
# 2. Runtime behavior — does it actually skip / perform the Redis write?
# ---------------------------------------------------------------------------


class _FakeAgent:
    name = "test-agent"


class _FakeCtx:
    run_id = "run-001"
    session_id = "sess-abc"
    user_id = "user-42"
    max_turns = 5
    trace_id = "trace-xyz"


def _svc_with_mock_manager(persist_state: bool) -> tuple[ContextService, MagicMock]:
    manager = MagicMock()
    manager.save = AsyncMock()
    svc = ContextService(
        state_manager=manager,
        config=RunnerConfig(persist_state=persist_state),
    )
    return svc, manager


class TestPersistStateRuntime:
    @pytest.mark.asyncio
    async def test_off_does_not_write(self):
        # persist_state=False: create_run_state must NOT touch the state manager.
        svc, manager = _svc_with_mock_manager(persist_state=False)
        await svc.create_run_state(_FakeAgent(), _FakeCtx())
        manager.save.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_on_writes(self):
        # persist_state=True: create_run_state must save exactly once.
        svc, manager = _svc_with_mock_manager(persist_state=True)
        await svc.create_run_state(_FakeAgent(), _FakeCtx())
        manager.save.assert_awaited_once()


# ---------------------------------------------------------------------------
# 3. Ticket regression — no Redis connection attempt / no warning by default
# ---------------------------------------------------------------------------


class TestTicketRegression:
    @pytest.mark.asyncio
    async def test_default_never_builds_state_manager(self, monkeypatch):
        # The ticket's root cause: run-state persistence reached for Redis on the
        # run path. With the default (disabled) config, the global state manager
        # must never be constructed — no manager, no Redis connection attempt.
        import continuum.agent.services.context_service as cs

        def _tripwire():
            raise AssertionError(
                "get_global_state_manager() must not be called when persist_state is False"
            )

        monkeypatch.setattr(cs, "get_global_state_manager", _tripwire)

        svc = ContextService(config=RunnerConfig())  # default -> persistence off
        state = await svc.create_run_state(_FakeAgent(), _FakeCtx())
        await svc.save_run_state(state)  # tripwire raises if either path touches it

    @pytest.mark.asyncio
    async def test_default_emits_no_redis_warning(self, caplog):
        # The ticket's observable symptom: a "Failed to connect to Redis for state
        # persistence" warning per run. With the default config it must not appear
        # (regardless of whether Redis is reachable), since the path is never taken.
        svc = ContextService(config=RunnerConfig())  # default -> persistence off
        with caplog.at_level("WARNING"):
            state = await svc.create_run_state(_FakeAgent(), _FakeCtx())
            await svc.save_run_state(state)
        assert not any("state persistence" in r.getMessage().lower() for r in caplog.records)
