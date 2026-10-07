"""Choosing a backend: by prefix, from the container, or from settings.

Configuring a backend enables nothing by itself -- that is each seam's switch.
What this layer guarantees is that a seam which HAS been switched on either
gets a backend or fails loudly, never quietly falls back to the old path.
"""

from __future__ import annotations

import pytest


class TestBackendsAreChosenByPrefix:
    def test_a_registered_prefix_builds_its_backend(self, fake_classifier_cls):
        from continuum.system_one import create_classifier, register_backend

        register_backend("fakeprefix", lambda model: fake_classifier_cls(model=model))
        backend = create_classifier("fakeprefix:model-a")
        assert backend.model == "model-a"

    def test_an_unknown_prefix_is_a_configuration_error(self):
        from continuum.exceptions import ConfigurationError
        from continuum.system_one import create_classifier

        with pytest.raises(ConfigurationError):
            create_classifier("nosuchbackend:x")

    def test_a_spec_without_a_model_is_a_configuration_error(self):
        from continuum.exceptions import ConfigurationError
        from continuum.system_one import create_classifier

        with pytest.raises(ConfigurationError):
            create_classifier("jev")

    def test_there_is_no_llm_prefix(self):
        """A general-purpose LLM is never a System One backend: the seams'
        existing LLM path already covers it."""
        from continuum.exceptions import ConfigurationError
        from continuum.system_one import create_classifier

        with pytest.raises(ConfigurationError):
            create_classifier("llm:gpt-4o-mini")

    def test_the_same_spec_returns_the_same_instance(self, fake_classifier_cls):
        """A local model is expensive to load; building it per call would be a
        per-request model load."""
        from continuum.system_one import register_backend, resolve_classifier

        register_backend("fakeprefix", lambda model: fake_classifier_cls(model=model))
        assert resolve_classifier("fakeprefix:m") is resolve_classifier("fakeprefix:m")

    def test_third_party_backends_load_through_entry_points(self, fake_classifier_cls, monkeypatch):
        """An open-source project plugs in without a change to the SDK."""
        import continuum.system_one.registry as registry

        class EP:
            name = "thirdparty"

            def load(self):
                return lambda model: fake_classifier_cls(model=model)

        def entry_points(*, group):
            assert group == "continuum.system_one"
            return [EP()]

        monkeypatch.setattr(registry, "entry_points", entry_points)
        backend = registry.create_classifier("thirdparty:m1")
        assert backend.model == "m1"


class TestResolvingTheBackendForASeam:
    def test_nothing_configured_is_a_configuration_error(self):
        from continuum.system_one import SystemOneNotConfiguredError, resolve_classifier

        with pytest.raises(SystemOneNotConfiguredError):
            resolve_classifier()

    def test_the_container_supplies_the_default(self, fake_classifier_cls, container):
        from continuum.system_one import resolve_classifier

        backend = fake_classifier_cls()
        container.set_system_one_classifier(backend)
        assert resolve_classifier() is backend

    def test_settings_supply_the_default_when_the_container_has_none(
        self, fake_classifier_cls, monkeypatch
    ):
        from continuum.config import settings
        from continuum.system_one import register_backend, resolve_classifier

        register_backend("fakeprefix", lambda model: fake_classifier_cls(model=model))
        monkeypatch.setattr(settings, "system_one_backend", "fakeprefix:from-env")
        assert resolve_classifier().model == "from-env"

    def test_a_seams_own_spec_overrides_the_default(self, fake_classifier_cls, container):
        from continuum.system_one import register_backend, resolve_classifier

        register_backend("fakeprefix", lambda model: fake_classifier_cls(model=model))
        container.set_system_one_classifier(fake_classifier_cls(model="default"))
        assert resolve_classifier("fakeprefix:override").model == "override"


class TestRequiringABackendAtConstruction:
    def test_it_raises_when_nothing_is_configured(self):
        from continuum.system_one import SystemOneNotConfiguredError, require_backend

        with pytest.raises(SystemOneNotConfiguredError) as exc:
            require_backend(None, seam="RouterAgent 'triage'")
        assert "RouterAgent 'triage'" in str(exc.value)

    def test_it_passes_with_an_explicit_spec(self):
        from continuum.system_one import require_backend

        require_backend("jev:jev-latest", seam="x")

    def test_it_passes_with_a_settings_default(self, monkeypatch):
        from continuum.config import settings
        from continuum.system_one import require_backend

        monkeypatch.setattr(settings, "system_one_backend", "jev:jev-latest")
        require_backend(None, seam="x")

    def test_it_passes_with_a_container_backend(self, fake_classifier_cls, container):
        from continuum.system_one import require_backend

        container.set_system_one_classifier(fake_classifier_cls())
        require_backend(None, seam="x")

    def test_it_does_not_build_the_backend(self, monkeypatch):
        """A construction check that loaded a local model or opened a client
        would make building an agent slow and network-dependent."""
        from continuum.config import settings
        from continuum.system_one import register_backend, require_backend

        built = []
        register_backend("fakeprefix", lambda model: built.append(model))
        monkeypatch.setattr(settings, "system_one_backend", "fakeprefix:m")
        require_backend(None, seam="x")
        assert built == []
