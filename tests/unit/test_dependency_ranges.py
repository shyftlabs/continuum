"""Every dependency Continuum declares has a range, not just a floor.

Continuum is a library: it states the versions it works with, and pip picks
one that also suits the user's other packages. A bare ``>=`` promised that every
future release would work. ``mem0ai>=1.0.0`` met mem0ai 2.0 and broke installs;
``openai>=1.50.0`` and ``pymilvus>=2.4.0`` have each since crossed majors the
SDK was never tested on. So each requirement now ends below the next major --
the next minor for a 0.x package, whose minors may break -- and is raised by a
deliberate PR when a new major is tested. Exact pins are allowed only for
tooling where everyone must agree (ruff), never for a runtime dependency.

The floors are checked separately: CI installs the lowest versions these ranges
allow and runs the unit tests (``tests-lowest`` in ci.yml).

``requirements.txt`` repeats the runtime ranges by hand, and had drifted (no
``httpx``, ``mcp`` uncapped); it must match ``pyproject.toml``.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]

# Tooling pinned exactly on purpose: everyone formats and lints identically.
EXACT_PINS_ALLOWED = {"ruff"}


def _requirements() -> list[tuple[str, Requirement]]:
    reqs = [("dependencies", Requirement(r)) for r in PROJECT["dependencies"]]
    for extra, items in PROJECT.get("optional-dependencies", {}).items():
        reqs += [(f"[{extra}]", Requirement(r)) for r in items]
    return reqs


@pytest.mark.parametrize(
    ("where", "req"), _requirements(), ids=lambda v: str(v) if isinstance(v, Requirement) else v
)
def test_every_requirement_has_a_floor_and_a_ceiling(where, req):
    ops = {spec.operator for spec in req.specifier}
    if req.name in EXACT_PINS_ALLOWED:
        assert ops == {"=="}, f"{where} {req}: expected an exact pin"
        return
    assert "==" not in ops, f"{where} {req}: a library must not pin a dependency exactly"
    assert ">=" in ops, f"{where} {req}: no lower bound"
    assert ops & {"<", "<="}, f"{where} {req}: no upper bound -- cap it below the next major"


def _requirements_txt() -> dict[str, str]:
    """The active (uncommented) lines of requirements.txt, name -> specifier."""
    out = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            req = Requirement(line)
            out[req.name] = str(req.specifier)
    return out


def test_requirements_txt_matches_pyproject():
    declared = {r.name: str(r.specifier) for r in map(Requirement, PROJECT["dependencies"])}
    assert _requirements_txt() == declared
