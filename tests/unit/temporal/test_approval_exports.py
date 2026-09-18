"""``temporal_tool_approval`` is reachable, and only with the extra installed.

An export looks too small to test, and one half of it is: a wrong import line
fails on first import.

The half worth testing is invisible in this environment. ``approval_adapter``
imports no ``temporalio`` at module level -- it reaches for it inside functions
-- so the name could sit in this package's PURE section and import fine without
the ``[temporal]`` extra. It would then build a handler that defers every
approval forever, because ``activity.info()`` raises and there is no workflow id
to resolve: a control that silently does nothing, wearing the name of one.

The test environment always has ``temporalio`` installed, so that path is one no
other test reaches and one that would only ever fail for a user.

WHY THE SIMULATION RUNS IN A SUBPROCESS

The first version blocked ``temporalio`` by swapping ``builtins.__import__`` and
popping ``sys.modules``, then restoring both. It worked in isolation and broke
ten unrelated tests, because the restore left ``continuum.temporal.*`` in a
state later tests inherited -- and the full suite went from 27s to 154s.

A subprocess cannot leak, and it is also the more faithful simulation: a user
without the extra has a fresh interpreter that never imported temporalio, not
one where it was removed halfway through.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _run_without_temporalio(body: str) -> subprocess.CompletedProcess:
    """Run `body` in a fresh interpreter where importing temporalio fails."""
    script = textwrap.dedent(f"""
        import sys

        class _Blocked:
            def find_module(self, name, path=None):
                return self if name.split(".")[0] == "temporalio" else None

            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] == "temporalio":
                    raise ImportError("No module named 'temporalio'")
                return None

        sys.meta_path.insert(0, _Blocked())
        for m in [m for m in sys.modules if m.split(".")[0] == "temporalio"]:
            del sys.modules[m]

        {textwrap.indent(textwrap.dedent(body), " " * 8).strip()}
    """)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=120,
    )


class TestTheNameIsPartOfThePublicApi:
    def test_it_is_importable_from_the_package(self):
        pytest.importorskip("temporalio", reason="the temporal extra is not installed")
        from continuum.temporal import temporal_tool_approval

        assert callable(temporal_tool_approval)

    def test_it_is_in_dunder_all(self):
        """Otherwise `from continuum.temporal import *` misses it, and it reads
        as an internal helper rather than the documented way in."""
        pytest.importorskip("temporalio", reason="the temporal extra is not installed")
        import continuum.temporal as mod

        assert "temporal_tool_approval" in mod.__all__
        assert "temporal_approval_handler" in mod.__all__

    def test_every_exported_name_actually_resolves(self):
        """__all__ is a promise. A name listed but never imported fails only at
        `import *`, which nothing else here does."""
        pytest.importorskip("temporalio", reason="the temporal extra is not installed")
        import continuum.temporal as mod

        missing = [n for n in mod.__all__ if not hasattr(mod, n)]
        assert not missing, f"__all__ names nothing imports: {missing}"


class TestWithoutTheTemporalExtra:
    """The path this environment cannot otherwise reach."""

    def test_the_package_still_imports(self):
        """The pure half -- types, config, exceptions, is_authorized -- stays
        usable without the runtime. That is the existing contract."""
        r = _run_without_temporalio("""
            import continuum.temporal as mod
            assert hasattr(mod, "ApprovalDecision"), "types went missing"
            assert hasattr(mod, "TemporalConfig"), "config went missing"
            print("OK")
        """)
        assert r.returncode == 0, f"importing without the extra failed:\n{r.stderr[-2000:]}"
        assert "OK" in r.stdout

    def test_the_approval_handler_is_absent_rather_than_inert(self):
        """Not exported without the extra, on purpose. approval_adapter has no
        module-level temporalio import, so it WOULD import cleanly here -- and
        then defer every approval forever, since there is no activity to read a
        workflow id from. A missing name and an ImportError naming the extra
        beats a gate that silently does nothing.
        """
        r = _run_without_temporalio("""
            import continuum.temporal as mod
            if hasattr(mod, "temporal_tool_approval"):
                raise SystemExit(
                    "temporal_tool_approval is reachable without the [temporal] extra — "
                    "it would build a handler that defers every approval and never asks "
                    "anyone. Move the import into the guarded block."
                )
            print("OK")
        """)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "OK" in r.stdout
