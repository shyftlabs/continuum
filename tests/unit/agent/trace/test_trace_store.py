"""
Unit tests for the infra-free trace stores (Slice 2).

The Redis backend is covered against *real* Redis in
tests/integration/test_trace_store_redis.py — no fakes here.
"""

from __future__ import annotations

from continuum.agent.trace import DecisionStep, DecisionTrace, StepKind
from continuum.agent.trace.store import (
    InMemoryTraceStore,
    NullTraceStore,
    RedisTraceStore,
    TraceStore,
)


def _trace(run_id: str = "run_x") -> DecisionTrace:
    t = DecisionTrace(run_id=run_id, root_agent="a", user_query="q")
    t.add(
        DecisionStep(
            step_id="s1", kind=StepKind.LLM_CALL, agent_name="a", output="hello", total_tokens=10
        )
    )
    t.final_response = "hello"
    return t


class TestProtocolConformance:
    def test_all_backends_satisfy_protocol(self) -> None:
        assert isinstance(NullTraceStore(), TraceStore)
        assert isinstance(InMemoryTraceStore(), TraceStore)
        # Constructing with no client does not connect (redis is lazy), so this is
        # safe offline and still asserts the real backend conforms.
        assert isinstance(RedisTraceStore(), TraceStore)


class TestInMemoryTraceStore:
    async def test_roundtrip_and_delete(self) -> None:
        store = InMemoryTraceStore()
        await store.save(_trace("r"))
        got = await store.get("r")
        assert got is not None
        assert got.run_id == "r"
        assert await store.delete("r") is True
        assert await store.get("r") is None
        assert await store.delete("r") is False


class TestNullTraceStore:
    async def test_noop(self) -> None:
        store = NullTraceStore()
        await store.save(_trace("r"))
        assert await store.get("r") is None
        assert await store.delete("r") is False


class TestRedisTraceStore:
    async def test_save_uses_supported_set_with_expiry(self) -> None:
        class FakeRedis:
            def __init__(self) -> None:
                self.calls = []

            async def set(self, key, value, *, ex):
                self.calls.append((key, value, ex))

            async def setex(self, *args, **kwargs):
                raise AssertionError("deprecated setex must not be used")

        redis = FakeRedis()
        store = RedisTraceStore(client=redis, ttl_seconds=90)

        await store.save(_trace("redis-run"))

        assert len(redis.calls) == 1
        assert redis.calls[0][0].endswith(":redis-run")
        assert redis.calls[0][2] == 90
