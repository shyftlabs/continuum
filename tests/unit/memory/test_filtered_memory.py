"""``pre_store_filter`` as a gate before the write, not a delete after it.

mem0 fuses extraction and storage inside one ``add()`` call and exposes no
extract-without-store path, so the filter used to run on what had already been
written and could only delete the rejects. Measured against a live Milvus that
delete races write visibility and loses often: a rejected SSN fact was still
searchable minutes later.

``_create_memory`` is the seam. It is a separate method, every write path funnels
through it (the ``infer=True`` ADD branch and the ``infer=False`` path both), and
Continuum uses mem0's sync ``Memory``, so one override covers all of it.
Suppressing there means the row is never inserted: nothing to delete, no race.

The cost is a dependency on two private methods, which is why
``assert_mem0_seam_intact`` exists — a rename upstream must fail loudly rather
than silently disable filtering.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# The tripwire — the private API this depends on is still shaped as expected.
# ---------------------------------------------------------------------------


class TestTheMem0SeamIsStillThere:
    def test_the_seam_is_intact_against_the_installed_mem0(self):
        from continuum.memory.providers.filtered_memory import assert_mem0_seam_intact

        assert_mem0_seam_intact()  # must not raise on the pinned version

    def test_a_renamed_method_is_reported_loudly(self):
        """Silently reverting to no filtering is the failure nobody notices."""
        from continuum.memory.providers.filtered_memory import (
            Mem0SeamError,
            assert_mem0_seam_intact,
        )

        class Renamed:
            def _update_memory(self, memory_id, data, existing_embeddings, metadata=None): ...

        with pytest.raises(Mem0SeamError, match="_create_memory"):
            assert_mem0_seam_intact(Renamed)

    def test_a_changed_signature_is_reported_loudly(self):
        from continuum.memory.providers.filtered_memory import (
            Mem0SeamError,
            assert_mem0_seam_intact,
        )

        class Changed:
            def _create_memory(self, payload, embeddings):  # `data` renamed away
                ...

            def _update_memory(self, memory_id, data, existing_embeddings, metadata=None): ...

        with pytest.raises(Mem0SeamError, match="data"):
            assert_mem0_seam_intact(Changed)


# ---------------------------------------------------------------------------
# The gate itself. FilteredMemory subclasses mem0's Memory, whose __init__ needs
# a live config; these drive the overrides directly against a stub base.
# ---------------------------------------------------------------------------


def _memory(filter_fn):
    """A FilteredMemory whose base class records writes instead of making them."""
    from continuum.memory.providers.filtered_memory import FilteredMemory, use_pre_store_filter

    written: list[str] = []

    class Base:
        def _create_memory(self, data, existing_embeddings, metadata=None):
            written.append(data)
            return "new-id"

        def _update_memory(self, memory_id, data, existing_embeddings, metadata=None):
            written.append(data)
            return memory_id

    subject = type("Subject", (FilteredMemory, Base), {})()

    def use_filter(fn):
        return use_pre_store_filter(subject, fn)

    return subject, written, use_filter


class TestTheGate:
    def test_an_allowed_fact_is_written(self):
        subject, written, use_filter = _memory(None)
        with use_filter(lambda facts: facts) as suppressed:
            assert subject._create_memory("morning appointments", {}) == "new-id"
        assert written == ["morning appointments"]
        assert suppressed == []

    def test_a_rejected_fact_is_never_written(self):
        subject, written, use_filter = _memory(None)
        with use_filter(lambda facts: [f for f in facts if "SSN" not in f]) as suppressed:
            result = subject._create_memory("SSN is 123-45-6789", {})
        assert written == [], "the row must never reach the store"
        assert result is None
        assert suppressed == ["SSN is 123-45-6789"]

    def test_a_raising_filter_rejects_the_fact(self):
        """Fail closed: a filter that cannot answer has said nothing about it."""
        subject, written, use_filter = _memory(None)

        def broken(_facts):
            raise RuntimeError("PII scanner unavailable")

        with use_filter(broken) as suppressed:
            assert subject._create_memory("anything at all", {}) is None
        assert written == []
        assert suppressed == ["anything at all"]

    def test_no_filter_means_no_gate(self):
        """The shipped default: no detector, no guesses, nothing examined."""
        subject, written, _ = _memory(None)
        assert subject._create_memory("stored verbatim", {}) == "new-id"
        assert written == ["stored verbatim"]

    def test_updates_are_gated_too(self):
        """Otherwise a rejected fact arrives by overwriting an existing row."""
        subject, written, use_filter = _memory(None)
        with use_filter(lambda facts: [f for f in facts if "SSN" not in f]) as suppressed:
            assert subject._update_memory("row-1", "SSN is 123-45-6789", {}) is None
        assert written == []
        assert suppressed == ["SSN is 123-45-6789"]

    def test_the_public_filter_signature_is_unchanged(self):
        """Filters are written as list[str] -> list[str] and keep working.

        The gate cannot know the batch (mem0 has not finished extracting), so it
        offers one fact at a time. A filter written against the documented
        signature must not notice.
        """
        seen: list[list[str]] = []

        def recording(facts: list[str]) -> list[str]:
            seen.append(facts)
            return facts

        subject, _, use_filter = _memory(None)
        with use_filter(recording):
            subject._create_memory("one", {})
            subject._create_memory("two", {})
        assert seen == [["one"], ["two"]]

    def test_the_filter_does_not_leak_past_its_scope(self):
        """A long-lived Memory instance serves every write; a filter set for one
        must not gate the next."""
        subject, written, use_filter = _memory(None)
        with use_filter(lambda _facts: []):
            subject._create_memory("rejected", {})
        subject._create_memory("later, unfiltered", {})
        assert written == ["later, unfiltered"]

    def test_concurrent_writes_do_not_share_a_filter(self):
        """One Memory instance serves every write, so a filter set for one must
        not gate another's facts.

        Driven from two threads because that is where the scope is entered:
        ``Mem0Provider`` opens it inside ``asyncio.to_thread``, never on the
        event loop. The lock is deliberately non-reentrant -- two scopes on one
        thread is the clobbering this prevents, not a case to support.
        """
        import threading

        subject, written, use_filter = _memory(None)
        start = threading.Barrier(2)

        def run(tag: str, reject: bool):
            start.wait(timeout=5)
            with use_filter((lambda _f: []) if reject else (lambda f: f)):
                subject._create_memory(tag, {})

        threads = [
            threading.Thread(target=run, args=("blocked", True)),
            threading.Thread(target=run, args=("allowed", False)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
            assert not t.is_alive(), "a write did not finish — the lock deadlocked"

        assert written == ["allowed"], "the rejecting filter leaked onto the other write"


class TestSuppressionIsReported:
    def test_suppressed_texts_are_collected_for_the_caller(self):
        """on_stored must not name a fact that was never written."""
        subject, _, use_filter = _memory(None)
        with use_filter(lambda facts: [f for f in facts if "no" not in f]) as suppressed:
            subject._create_memory("yes please", {})
            subject._create_memory("no thanks", {})
            subject._create_memory("no really", {})
        assert suppressed == ["no thanks", "no really"]

    def test_a_filter_returning_a_rewritten_string_does_not_count_as_allowed(self):
        """Only membership decides. A filter is a gate, not a transformer -- a
        returned-but-different string would otherwise store text the caller never
        approved under the guise of having passed."""
        subject, written, use_filter = _memory(None)
        with use_filter(lambda facts: [f.upper() for f in facts]) as suppressed:
            subject._create_memory("quiet", {})
        assert written == []
        assert suppressed == ["quiet"]


class TestTheClassHandedToMem0:
    def test_it_is_a_real_mem0_memory_with_the_gate_in_front(self):
        from mem0.memory.main import Memory

        from continuum.memory.providers.filtered_memory import (
            FilteredMemory,
            build_filtered_memory_class,
        )

        cls = build_filtered_memory_class()
        assert issubclass(cls, Memory), "mem0 must still see one of its own"
        assert issubclass(cls, FilteredMemory)
        # The gate has to come first in the MRO or super() dispatch writes the
        # row before anything is asked.
        assert cls.__mro__.index(FilteredMemory) < cls.__mro__.index(Memory)

    def test_from_config_constructs_the_gated_class(self):
        """mem0's from_config ends `return cls(config)`, so the subclass survives
        it. If that ever becomes `return Memory(config)` the gate disappears with
        no error, which is why this is asserted rather than assumed."""
        from continuum.memory.providers.filtered_memory import build_filtered_memory_class

        cls = build_filtered_memory_class()
        built = {}
        with pytest.MonkeyPatch.context() as mp:

            def _init(self, config):  # __init__ must return None
                built["cls"] = type(self)

            mp.setattr(cls, "__init__", _init)
            cls.from_config({"vector_store": {"provider": "qdrant"}})
        assert built["cls"] is cls

    def test_the_seam_is_checked_when_the_class_is_built(self):
        """The check must run on the path that actually builds the class, not
        only when a test calls it directly."""
        import continuum.memory.providers.filtered_memory as fm

        called = []
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(fm, "assert_mem0_seam_intact", lambda *a: called.append(a))
            fm.build_filtered_memory_class()
        assert called, "build_filtered_memory_class must verify the seam"


class TestTheGateCannotBeBypassedByConstruction:
    """Two changes the seam check cannot see, guarded here instead."""

    def test_the_provider_uses_the_synchronous_memory(self):
        """mem0's AsyncMemory has its own _create_memory that this mixin does
        not cover. Continuum runs the sync class through asyncio.to_thread; a
        switch to the async one would remove the gate with no error anywhere."""
        import inspect

        from continuum.memory.providers import mem0 as provider_mod

        src = inspect.getsource(provider_mod)
        assert "build_filtered_memory_class()" in src, (
            "the provider no longer builds the gated class"
        )
        # Usage, not mention: the module docstring names AsyncMemory as an
        # option, which is fine. Importing or constructing it is not.
        for forbidden in ("import AsyncMemory", "AsyncMemory(", "AsyncMemory.from_config"):
            assert forbidden not in src, (
                f"the provider uses AsyncMemory ({forbidden}), whose _create_memory is "
                f"not gated — extend FilteredMemory to cover it before switching"
            )

    def test_the_delete_fallback_is_still_wired(self):
        """The fallback is what keeps a bypassed write from being silent, so it
        is not dead code to be tidied away."""
        import inspect

        from continuum.session import client as session_client

        src = inspect.getsource(session_client.SessionClient._store_in_memory)
        assert "pre_store_filter(fact_texts)" in src, "the post-write filter fallback is gone"
        assert "ok is False" in src, "the failed-delete report is gone"
