"""``pre_store_filter`` must guard every write IntelligentMemoryClient makes.

Background
----------
One ``add()`` on the intelligent client makes up to three writes: the normal
fact write, one per extracted entity, and a user-profile update. Only the first
was handed the filter. The entity and profile writes called ``super().add()``
without it, so a fact the filter rejects -- an SSN, say -- was stopped on the
first write and stored by the second or third.

Nothing behind those writes catches it either. SessionClient's delete-after-write
fallback inspects the result of the fact write only; the entity and profile
results never leave this class. So a rejected fact written this way stayed.

Each extra write now checks the text it is about to store against the filter
before writing, with the same semantics as the provider gate (membership decides;
a filter that raises rejects), and also forwards the filter so whatever mem0
extracts from that text is gated too.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from continuum.memory.client import MemoryClient
from continuum.memory.config import MemoryConfig
from continuum.memory.intelligence import IntelligenceConfig, IntelligentMemoryClient

_SSN = "123-45-6789"


class _FakeLLM:
    """Answers the entity prompt and the profile prompt; both leak the SSN."""

    def __init__(self, profile: dict[str, Any] | None = None) -> None:
        self._profile = profile or {"employer": f"Acme, SSN {_SSN}"}

    async def chat(self, messages, **_kwargs):  # noqa: ANN001
        prompt = messages[0]["content"]
        if "Extract named entities" in prompt:
            body = {"entities": [{"name": "John", "type": "person", "attributes": {"ssn": _SSN}}]}
        else:
            body = self._profile
        return SimpleNamespace(content=json.dumps(body))


@pytest.fixture
def writes(monkeypatch) -> list[dict[str, Any]]:
    """Every call that reaches MemoryClient.add, i.e. every write."""
    calls: list[dict[str, Any]] = []

    async def _fake_add(self, messages, **kwargs):  # noqa: ANN001
        calls.append({"messages": messages, **kwargs})
        return SimpleNamespace(results=[])

    async def _no_sleep(_seconds):  # noqa: ANN001
        return None

    monkeypatch.setattr(MemoryClient, "add", _fake_add)
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    return calls


def _client(llm: _FakeLLM | None = None) -> IntelligentMemoryClient:
    client = IntelligentMemoryClient(
        config=MemoryConfig(enabled=True),
        intelligence_config=IntelligenceConfig(
            enable_scoring=False, enable_entity_memory=True, enable_user_profiles=True
        ),
    )
    client._get_llm = lambda: llm or _FakeLLM()  # type: ignore[method-assign]

    async def _no_profile(_user_id):  # noqa: ANN001
        return None

    client.get_user_profile = _no_profile  # type: ignore[method-assign]
    return client


def _kind(call: dict[str, Any]) -> str:
    return (call.get("metadata") or {}).get("memory_type", "fact")


def _no_ssn(facts: list[str]) -> list[str]:
    return [f for f in facts if _SSN not in f]


# ---------------------------------------------------------------------------
# The gap: the entity and profile writes ignored the filter
# ---------------------------------------------------------------------------


class TestARejectedFactIsNotStoredByTheExtraWrites:
    async def test_entity_carrying_a_rejected_fact_is_not_written(self, writes):
        await _client().add("I'm John", user_id="u1", pre_store_filter=_no_ssn)

        assert "entity" not in [_kind(c) for c in writes]

    async def test_profile_carrying_a_rejected_fact_is_not_written(self, writes):
        await _client().add("I'm John", user_id="u1", pre_store_filter=_no_ssn)

        assert "user_profile" not in [_kind(c) for c in writes]

    async def test_a_filter_that_raises_writes_neither(self, writes):
        """Fail closed, as the provider gate does: a filter that cannot answer
        has said nothing about the text."""

        def broken(_facts):  # noqa: ANN001
            raise RuntimeError("classifier down")

        await _client().add("I'm John", user_id="u1", pre_store_filter=broken)

        assert {_kind(c) for c in writes} <= {"fact"}

    async def test_a_rewritten_fact_counts_as_rejected(self, writes):
        """A filter is a gate, not a transformer -- the same rule the provider
        gate applies. Storing the original after a rewrite would store exactly
        what the filter tried to remove."""
        await _client().add(
            "I'm John",
            user_id="u1",
            pre_store_filter=lambda facts: [f.replace(_SSN, "***") for f in facts],
        )

        assert {_kind(c) for c in writes} <= {"fact"}


class TestTheFilterIsForwardedToEveryWrite:
    """So mem0's own extraction from the entity text is gated as well."""

    async def test_every_write_receives_the_filter(self, writes):
        def allow_all(facts):  # noqa: ANN001
            return facts

        await _client().add("I'm John", user_id="u1", pre_store_filter=allow_all)

        assert {_kind(c) for c in writes} == {"fact", "entity", "user_profile"}
        assert all(c.get("pre_store_filter") is allow_all for c in writes)


class TestTheProfileCannotSmuggleUnseenKeys:
    async def test_keys_outside_the_documented_set_are_not_stored(self, writes):
        """profile_json rides in metadata, which the filter never sees; only the
        summary text is offered to it. A key the summary does not render would
        be stored unchecked, so only the documented keys are kept."""
        llm = _FakeLLM(profile={"employer": "Acme", "ssn": _SSN})

        await _client(llm).add("I'm John", user_id="u1", pre_store_filter=_no_ssn)

        profile = [c for c in writes if _kind(c) == "user_profile"]
        assert profile, "an allowed profile update should still be written"
        assert "ssn" not in json.loads(profile[0]["metadata"]["profile_json"])


# ---------------------------------------------------------------------------
# What must not change
# ---------------------------------------------------------------------------


class TestWhatMustNotChange:
    async def test_without_a_filter_all_three_writes_happen(self, writes):
        await _client().add("I'm John", user_id="u1")

        assert {_kind(c) for c in writes} == {"fact", "entity", "user_profile"}

    async def test_an_allowed_entity_and_profile_are_written(self, writes):
        llm = _FakeLLM(profile={"employer": "Acme"})

        await _client(llm).add("I'm John", user_id="u1", pre_store_filter=lambda f: f)

        assert {_kind(c) for c in writes} == {"fact", "entity", "user_profile"}
