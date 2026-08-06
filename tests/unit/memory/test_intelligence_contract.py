"""
IntelligentMemoryClient must honour MemoryClient's method contract.

It documents itself as a "Drop-in replacement for MemoryClient. All existing
methods work identically", and callers hold it behind a `MemoryClient`-typed
reference. Its `add()`/`search()` overrides had dropped four keyword arguments
(`infer`, `policy_store`, `subject`, `data_labels`), so any caller passing them
would get a TypeError from what is documented as a transparent substitute.

The security consequence was checked and is narrower than it looks: the base
`add()` resolves the access-control gate through `resolve_active_policy`, which
falls back to the ambient run policy when the explicit args are absent — and the
subclass does call `super().add()`. So the gate was never bypassed. What was
genuinely unreachable through the subclass was `infer`, and any explicit
per-call policy override.

These tests pin the signatures rather than the behaviour, so they stay valid
without a live mem0/vector-store backend.
"""

from __future__ import annotations

import inspect

from continuum.memory.client import MemoryClient
from continuum.memory.intelligence import IntelligentMemoryClient

# Parameters the base class accepts that the subclass previously dropped.
_PREVIOUSLY_DROPPED = {
    "add": {"infer", "policy_store", "subject", "data_labels"},
    "search": {"policy_store", "subject"},
}


def _params(cls: type, method: str) -> dict[str, inspect.Parameter]:
    return dict(inspect.signature(getattr(cls, method)).parameters)


class TestSignatureCompatibility:
    """Every base parameter must survive on the override, with the same default."""

    def test_add_accepts_every_base_parameter(self) -> None:
        base, sub = _params(MemoryClient, "add"), _params(IntelligentMemoryClient, "add")
        missing = set(base) - set(sub)
        assert not missing, f"IntelligentMemoryClient.add drops {sorted(missing)}"

    def test_search_accepts_every_base_parameter(self) -> None:
        base, sub = _params(MemoryClient, "search"), _params(IntelligentMemoryClient, "search")
        missing = set(base) - set(sub)
        assert not missing, f"IntelligentMemoryClient.search drops {sorted(missing)}"

    def test_defaults_match_the_base(self) -> None:
        """A differing default would silently change behaviour for a caller that
        believes it is talking to a MemoryClient."""
        for method in ("add", "search"):
            base, sub = _params(MemoryClient, method), _params(IntelligentMemoryClient, method)
            for name, base_param in base.items():
                if name == "self":
                    continue
                assert sub[name].default == base_param.default, (
                    f"{method}(): default for '{name}' differs from MemoryClient"
                )

    def test_the_specific_parameters_that_regressed(self) -> None:
        """Named explicitly so the regression is legible without a signature diff."""
        for method, dropped in _PREVIOUSLY_DROPPED.items():
            sub = _params(IntelligentMemoryClient, method)
            for name in dropped:
                assert name in sub, f"{method}() is missing '{name}'"


class TestPolicyArgumentsAreForwarded:
    """Accepting the arguments is not enough — they must reach the base call,
    or an explicit per-call policy override would be silently ignored, which is
    worse than the TypeError it replaces."""

    async def test_add_forwards_policy_arguments(self, monkeypatch) -> None:
        captured: dict[str, object] = {}

        async def _fake_add(self, messages, **kwargs):  # noqa: ANN001
            captured.update(kwargs)
            return None

        monkeypatch.setattr(MemoryClient, "add", _fake_add)
        client = IntelligentMemoryClient.__new__(IntelligentMemoryClient)
        from continuum.memory.intelligence import IntelligenceConfig

        client._intel = IntelligenceConfig(
            enable_scoring=False, enable_entity_memory=False, enable_user_profiles=False
        )

        sentinel_store, labels = object(), {"pii"}
        await client.add(
            "hello",
            user_id="u1",
            infer=False,
            policy_store=sentinel_store,
            subject="agent-x",
            data_labels=labels,
        )

        assert captured["infer"] is False
        assert captured["policy_store"] is sentinel_store
        assert captured["subject"] == "agent-x"
        assert captured["data_labels"] == labels

    async def test_search_forwards_policy_arguments(self, monkeypatch) -> None:
        captured: dict[str, object] = {}

        async def _fake_search(self, query, **kwargs):  # noqa: ANN001
            captured.update(kwargs)
            from continuum.memory.types import MemorySearchResult

            return MemorySearchResult(results=[], query=query, limit=10)

        monkeypatch.setattr(MemoryClient, "search", _fake_search)
        client = IntelligentMemoryClient.__new__(IntelligentMemoryClient)
        from continuum.memory.intelligence import IntelligenceConfig

        client._intel = IntelligenceConfig(enable_scoring=False, enable_decay=False)

        sentinel_store = object()
        await client.search("q", user_id="u1", policy_store=sentinel_store, subject="agent-x")

        assert captured["policy_store"] is sentinel_store
        assert captured["subject"] == "agent-x"
