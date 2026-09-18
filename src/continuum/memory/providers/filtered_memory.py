"""A ``pre_store_filter`` that runs *before* the write.

mem0 fuses fact extraction and storage inside one ``Memory.add()`` call: the
extraction is inline in ``_add_to_vector_store``, not a method, and there is no
extract-without-store path in 1.x. So a filter applied by the caller can only
run on what has already been persisted, and rejecting a fact means *deleting* it
rather than vetoing it.

Against Milvus that delete loses a race it cannot win. mem0's ``delete()`` reads
the row back before removing it (it needs the old value for its history log),
and Milvus defaults to ``Bounded`` consistency -- recent writes are deliberately
invisible to reads for a staleness window. Measured live: the delete issued
milliseconds after the write failed with ``IndexError`` from mem0's Milvus
adapter, and the rejected SSN fact was still searchable minutes later.

``_create_memory`` is the seam that fixes this. It is a real method, every write
funnels through it -- the ``infer=True`` ADD branch and the ``infer=False`` path
both call it -- and Continuum uses mem0's synchronous ``Memory``, so one override
covers every route. Suppressing there means the row is never inserted: nothing to
delete, no race, no window.

The price is a dependency on two private methods, and mem0 is pinned only to
``>=1.0.0,<2.0.0`` -- a fresh install can resolve to a minor version this was
never run against.

WHAT THE TRIPWIRE COVERS, AND WHAT IT DOES NOT

``assert_mem0_seam_intact`` fails construction if ``_create_memory`` or
``_update_memory`` is renamed or loses the argument the gate reads. That is the
likely breakage and the one worth an exception, because the alternative is a
filter that silently stops filtering while every log line still says it is
configured.

It cannot see two other changes:

* **A write path that bypasses these methods.** If a future mem0 inserts rows
  directly -- a batch write, a new branch in ``_add_to_vector_store`` -- the
  check still passes while those facts are never offered to the filter. This is
  why ``SessionClient`` keeps the old delete-and-report path: a fact that
  reaches the store anyway is still rejected, still deleted, and still reported
  at ERROR when the delete fails. Protection degrades to what it was before this
  module existed, which is weaker but not silent.
* **A switch to mem0's ``AsyncMemory``.** It has its own ``_create_memory``
  (main.py, the async half of the file) which this mixin does not cover.
  ``Mem0Provider`` builds the synchronous ``Memory`` and runs it through
  ``asyncio.to_thread``; if that ever changes, the gate disappears with no
  error. ``test_the_provider_uses_the_synchronous_memory`` is the guard.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import threading
from collections.abc import Callable, Iterator
from typing import Any

logger = logging.getLogger(__name__)

PreStoreFilter = Callable[[list[str]], list[str]]

# The active filter and the texts it suppressed, for the duration of one write.
#
# This was a ContextVar, which is the obvious choice and does not work.
# `asyncio.to_thread` does copy the context into its worker, but mem0's `add()`
# then submits `_add_to_vector_store` to a ThreadPoolExecutor of its own
# (main.py: `executor.submit(self._add_to_vector_store, ...)`), and a plain
# `submit()` starts the callable with a fresh context. Measured: the value set
# in the caller reads back as None inside the worker. So the filter never
# reached `_create_memory`, the gate silently did nothing, and the old
# delete-after-write path ran exactly as before -- the failure mode this module
# exists to remove, reproduced by its own first implementation.
#
# Instance state crosses that boundary because `self` does. The cost is that a
# Memory instance serves every write, so two concurrent writes would otherwise
# see each other's filter: `_FILTER_LOCK` serialises them, and only while a
# filter is actually configured. Memory writes run off the response path, so
# queueing them is a latency cost nobody is waiting on.
_FILTER_ATTR = "_continuum_active_filter"


class Mem0SeamError(RuntimeError):
    """mem0's private write methods are not shaped the way the gate expects."""


_EXPECTED = {
    "_create_memory": ("data",),
    "_update_memory": ("memory_id", "data"),
}


def assert_mem0_seam_intact(cls: type | None = None) -> None:
    """Fail loudly if the methods the gate overrides have moved or changed.

    Called at construction rather than import so the error names the running
    configuration. What it protects against is not a crash -- an upstream rename
    would leave ``FilteredMemory`` defining methods nothing calls, so every write
    would proceed unfiltered while the configuration still claimed a filter.
    """
    if cls is None:  # pragma: no cover - exercised via the public path
        from mem0.memory.main import Memory

        cls = Memory

    for name, required in _EXPECTED.items():
        method = getattr(cls, name, None)
        if method is None:
            raise Mem0SeamError(
                f"mem0's {cls.__name__}.{name} is missing, so pre_store_filter cannot "
                f"gate writes before they happen. Pin mem0 to a version that has it, "
                f"or drop the filter rather than run with one that does nothing."
            )
        params = inspect.signature(method).parameters
        missing = [p for p in required if p not in params]
        if missing:
            raise Mem0SeamError(
                f"mem0's {cls.__name__}.{name} no longer takes {', '.join(missing)} "
                f"(now: {', '.join(params)}). The pre_store_filter gate reads that "
                f"argument to decide whether to write; it cannot be trusted as-is."
            )


@contextlib.contextmanager
def use_pre_store_filter(memory: Any, filter_fn: PreStoreFilter | None) -> Iterator[list[str]]:
    """Make ``filter_fn`` the active gate on ``memory`` for the enclosing block.

    Yields the list that suppressed fact texts are appended to, so the caller can
    tell what was withheld. Without that the caller would report a suppressed
    fact as stored: mem0's ADD branch appends ``{"id": None, "memory": text}``
    whatever ``_create_memory`` returns.

    Holds the instance lock for the duration, so two writes cannot see each
    other's filter. Uncontended when no filter is configured, which is the
    shipped default.

    Must be entered on the thread that runs mem0, not on the event loop: a
    non-reentrant lock taken in the loop thread would either deadlock the loop
    or, if made reentrant, let two interleaved coroutines clobber each other's
    filter -- which is the bug it exists to prevent.
    """
    suppressed: list[str] = []
    if filter_fn is None:
        yield suppressed
        return

    lock = getattr(memory, "_continuum_filter_lock", None)
    if lock is None:  # a stub in a unit test, or a Memory built before the mixin
        lock = threading.Lock()
    with lock:
        setattr(memory, _FILTER_ATTR, (filter_fn, suppressed))
        try:
            yield suppressed
        finally:
            setattr(memory, _FILTER_ATTR, None)


def _allowed(memory: Any, data: str) -> bool:
    """Ask the active filter about one fact. No filter means no gate."""
    active = getattr(memory, _FILTER_ATTR, None)
    if active is None:
        return True
    filter_fn, suppressed = active

    try:
        kept = filter_fn([data])
    except Exception as e:
        # Fail closed. A filter that raised has said nothing about this fact, and
        # keeping it would lose the guarantee the filter was added to provide.
        logger.error(
            "pre_store_filter raised (%s: %s) — the fact was NOT written, since "
            "nothing is known about its contents",
            type(e).__name__,
            e,
        )
        suppressed.append(data)
        return False

    # Membership, not truthiness: a filter is a gate, not a transformer. A
    # returned-but-rewritten string would otherwise store text the caller never
    # approved while appearing to have passed.
    if any(k == data for k in kept or []):
        return True

    # Text is deliberately not logged. It is what the filter exists to keep out
    # of a persistent store, and a log is usually the less guarded of the two.
    logger.info("🚫 pre_store_filter suppressed a fact before the write")
    suppressed.append(data)
    return False


class FilteredMemory:
    """Mixin over mem0's ``Memory`` that gates writes on the active filter.

    A mixin rather than a direct subclass so the gate can be tested against a
    stub base: ``Memory.__init__`` builds a live vector store, an embedder and an
    LLM, none of which a unit test should need to decide whether one string is
    allowed through.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Set before super().__init__ so the attributes exist even if mem0's
        # constructor raises and something later inspects the half-built object.
        self._continuum_filter_lock = threading.Lock()
        setattr(self, _FILTER_ATTR, None)
        super().__init__(*args, **kwargs)

    def _create_memory(
        self, data: str, existing_embeddings: Any, metadata: Any = None
    ) -> str | None:
        if not _allowed(self, data):
            return None
        return super()._create_memory(data, existing_embeddings, metadata)  # type: ignore[misc]

    def _update_memory(
        self, memory_id: Any, data: str, existing_embeddings: Any, metadata: Any = None
    ) -> Any:
        # Gated too: without this a rejected fact arrives by overwriting a row
        # that already exists, which the create-path gate would never see.
        if not _allowed(self, data):
            return None
        return super()._update_memory(memory_id, data, existing_embeddings, metadata)  # type: ignore[misc]


def build_filtered_memory_class() -> type:
    """``Memory`` with the gate mixed in, checked against the installed mem0.

    Built on demand rather than at import so a deployment that never enables
    memory does not pay for mem0's import, and so the seam check runs where its
    error can name the configuration that asked for it.
    """
    from mem0.memory.main import Memory

    assert_mem0_seam_intact(Memory)
    return type("GatedMemory", (FilteredMemory, Memory), {})
