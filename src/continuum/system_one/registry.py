"""Choosing a System One backend: by spec prefix, container, or settings.

A spec is ``<prefix>:<model>`` -- ``jev:jev-latest``, ``local:<hf-model-id>``.
Built-in prefixes are registered here; third-party adapters register through the
``continuum.system_one`` entry-point group, so an open-source project plugs in
with no change to the SDK::

    [project.entry-points."continuum.system_one"]
    mybackend = "my_package:make_classifier"   # called as make_classifier(model)

There is deliberately no ``llm:`` prefix. A general-purpose LLM is not a System
One backend: with a seam's switch off, that seam already uses the LLM.

Configuring a backend enables nothing. Each seam opts in on its own; what this
module guarantees is that a seam which HAS opted in either gets a backend or
fails loudly (:func:`require_backend`, :func:`resolve_classifier`).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from importlib import import_module
from importlib.metadata import entry_points
from typing import cast

from continuum.config import settings
from continuum.logging import get_logger
from continuum.protocols import ISystemOneClassifier
from continuum.system_one.exceptions import SystemOneNotConfiguredError

logger = get_logger(__name__)

ENTRY_POINT_GROUP = "continuum.system_one"

BackendFactory = Callable[[str], ISystemOneClassifier]

_factories: dict[str, BackendFactory] = {}
_instances: dict[str, ISystemOneClassifier] = {}
_lock = threading.Lock()


def _builtin(module: str, cls: str) -> BackendFactory:
    # Imported on first use: the local backend pulls in sentence-transformers,
    # which a Jev-only deployment should never have to install.
    def factory(model: str) -> ISystemOneClassifier:
        backend = getattr(import_module(f"continuum.system_one.backends.{module}"), cls)
        return cast(ISystemOneClassifier, backend(model=model))

    return factory


_BUILTIN: dict[str, BackendFactory] = {
    "jev": _builtin("jev", "JevClassifier"),
    "local": _builtin("local_nli", "LocalNLIClassifier"),
    "laya": _builtin("laya", "LayaClassifier"),
    "laya-mlx": _builtin("laya", "LayaMLXClassifier"),
}


def register_backend(prefix: str, factory: BackendFactory) -> None:
    """Register ``factory`` for specs starting ``<prefix>:``. Replaces any
    earlier registration, and drops cached instances built by the old one."""
    with _lock:
        _factories[prefix] = factory
        for spec in [s for s in _instances if s.split(":", 1)[0] == prefix]:
            del _instances[spec]


def clear_classifier_cache() -> None:
    """Forget cached instances and runtime registrations (for tests)."""
    with _lock:
        _instances.clear()
        _factories.clear()


def _parse(spec: str) -> tuple[str, str]:
    prefix, sep, model = spec.partition(":")
    if not sep or not prefix.strip() or not model.strip():
        raise SystemOneNotConfiguredError(
            f"System One backend spec {spec!r} must look like '<backend>:<model>', "
            "e.g. 'jev:jev-latest' or 'local:cross-encoder/nli-deberta-v3-small'.",
            config_key="system_one_backend",
        )
    return prefix.strip(), model.strip()


def _factory_for(prefix: str) -> BackendFactory:
    with _lock:
        factory = _factories.get(prefix)
    if factory is not None:
        return factory
    if prefix in _BUILTIN:
        return _BUILTIN[prefix]
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name == prefix:
            loaded = cast(BackendFactory, ep.load())
            with _lock:
                _factories[prefix] = loaded
            return loaded
    hint = (
        " A general-purpose LLM is not a System One backend; leave the seam's "
        "switch off to use the LLM."
        if prefix == "llm"
        else ""
    )
    raise SystemOneNotConfiguredError(
        f"Unknown System One backend '{prefix}'. Built in: {sorted(_BUILTIN)}; others "
        f"register through the '{ENTRY_POINT_GROUP}' entry-point group.{hint}",
        config_key="system_one_backend",
    )


def create_classifier(spec: str) -> ISystemOneClassifier:
    """Build a new backend instance for ``spec`` (uncached)."""
    prefix, model = _parse(spec)
    return _factory_for(prefix)(model)


def _cached(spec: str) -> ISystemOneClassifier:
    with _lock:
        existing = _instances.get(spec)
    if existing is not None:
        return existing
    built = create_classifier(spec)
    with _lock:
        return _instances.setdefault(spec, built)


def resolve_classifier(spec: str | None = None) -> ISystemOneClassifier:
    """The backend a seam should use.

    ``spec`` (a seam's own override) wins; then the container's backend; then
    ``SYSTEM_ONE_BACKEND``. Instances are cached per spec: a local model is
    expensive to load, and building one per call would be a model load per
    request.
    """
    if spec:
        return _cached(spec)

    from continuum.core.container import get_container

    backend = get_container().system_one_classifier
    if backend is not None:
        return backend
    raise SystemOneNotConfiguredError(
        "No System One backend is configured. Set SYSTEM_ONE_BACKEND (e.g. "
        "'jev:jev-latest'), call container.set_system_one_classifier(...), or give "
        "the seam its own backend spec.",
        config_key="system_one_backend",
    )


def require_backend(spec: str | None, *, seam: str) -> None:
    """Fail at construction when ``seam`` opted in with no backend to answer it.

    Checks only that one is configured -- it builds nothing, so constructing an
    agent stays fast and offline. A seam switched on with nothing behind it is
    an error, not a silent fallback: otherwise you would believe the classifier
    is deciding when it is not.
    """
    if spec:
        _parse(spec)
        return
    if getattr(settings, "system_one_backend", None):
        return
    from continuum.core.container import get_container

    if get_container().has_system_one_classifier():
        return
    raise SystemOneNotConfiguredError(
        f"{seam} is set to use the System One classifier, but no backend is "
        "configured. Set SYSTEM_ONE_BACKEND (e.g. 'jev:jev-latest'), call "
        "container.set_system_one_classifier(...) before building the agent, or "
        "give the seam its own backend spec.",
        config_key="system_one_backend",
    )
