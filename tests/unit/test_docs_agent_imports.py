"""Every import docs/agent.md shows must work.

The debate, scatter and supervised sections imported their factories from
``continuum.agent``, but only ``continuum.agent.workflow`` exported them: a
reader copying the example got an ImportError on its first line, while the
router, sequential, parallel, loop, planner and reflection factories beside
them worked. Nothing checked the documented imports against the package.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

DOC = Path(__file__).resolve().parents[2] / "docs" / "agent.md"

_IMPORT = re.compile(r"^\s*from\s+(continuum[\w.]*)\s+import\s+(\([^)]*\)|[^\n#]+)", re.M)


def _documented_imports() -> list[tuple[str, str]]:
    found = []
    for match in _IMPORT.finditer(DOC.read_text()):
        module, names = match.group(1), match.group(2).strip("()")
        for name in names.replace("\n", " ").split(","):
            name = name.strip().split(" as ")[0].strip()
            if name and name not in ("...", "…"):
                found.append((module, name))
    return found


def test_the_doc_has_imports_to_check():
    """Guards the regex: a pattern that matched nothing would pass every test below."""
    assert len(_documented_imports()) > 10


@pytest.mark.parametrize(("module", "name"), _documented_imports(), ids=lambda v: v)
def test_documented_import_resolves(module: str, name: str):
    assert hasattr(importlib.import_module(module), name), f"from {module} import {name}"


@pytest.mark.parametrize(
    "name", ["create_debate_agent", "create_scatter_agent", "create_supervised_agent"]
)
def test_every_workflow_factory_is_exported_beside_its_siblings(name: str):
    import continuum.agent

    assert name in continuum.agent.__all__
