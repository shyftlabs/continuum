from __future__ import annotations

import json

from continuum.agent.persistence.state import RunStateManager
from continuum.agent.types import RunState


async def test_run_state_save_uses_supported_set_with_expiry() -> None:
    class FakeRedis:
        def __init__(self) -> None:
            self.set_calls = []

        def set(self, key, value, *, ex):
            self.set_calls.append((key, value, ex))

        def setex(self, *args, **kwargs):
            raise AssertionError("deprecated setex must not be used")

        def sadd(self, *args):
            return None

        def expire(self, *args):
            return None

    redis = FakeRedis()
    manager = RunStateManager(auto_initialize=False, state_ttl=120)
    manager._redis = redis
    manager._initialized = True

    await manager.save(RunState(run_id="run-1", session_id="session-1", user_id="user-1"))

    assert len(redis.set_calls) == 1
    assert redis.set_calls[0][0].endswith(":run-1")
    assert json.loads(redis.set_calls[0][1])["run_id"] == "run-1"
    assert redis.set_calls[0][2] == 120
