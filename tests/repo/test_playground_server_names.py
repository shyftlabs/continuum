"""Every tracked playground must name its MCP servers explicitly.

Without name=, the transports invent one from the URL or command
("streamable_http: http://localhost:8890/mcp"). That was a display label for
error messages, but with namespace_tools defaulting to True it becomes the
prefix on every LLM-facing tool name -- so tool identity carries the host and
port, and ~39 of the 64 characters providers allow are gone before the tool
name starts.

The SDK warns at runtime (MCPUtil._warn_if_server_name_is_derived). This is the
build-time half: an audit found 33 of 35 playground constructions missing
name=, so catching the next one needs a check that does not depend on anyone
running the playground and reading the logs.

Scoped to git-tracked files, via `git ls-files playground/...` -- so the audit's
reach comes from that glob, not from where this file sits. playground/local/**
is gitignored scratch space and stays invisible to it either way.

Lives in tests/repo/ rather than tests/unit/: it is a convention check over the
whole tree, not an SDK unit test, and CI's library gate (`pytest tests/unit`)
should not redden for a demo. CI runs this directory as its own step; locally,
plain `pytest` collects it (testpaths = ["tests"]).
"""

from __future__ import annotations

import ast
import inspect
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

MCP_SERVER_CTORS = {
    "MCPServerStreamableHttp",
    "MCPServerSse",
    "MCPServerStdio",
    "MCPServerFunction",
}


def _tracked_playground_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "playground/*.py", "playground/**/*.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [REPO_ROOT / line for line in out.stdout.splitlines() if line.strip()]


def _name_position(ctor: str) -> int | None:
    """Index of the `name` parameter, so a positionally-supplied name counts.

    MCPServerFunction takes name as its first required positional; the three
    transports take it third. Reading the real signature keeps this correct if
    those ever change, rather than silently reporting false positives.
    """
    import continuum.tools.mcp as mcp_mod

    params = list(inspect.signature(getattr(mcp_mod, ctor)).parameters.values())
    for i, p in enumerate(params):
        if p.name == "name":
            return i
    return None


def _unnamed_constructions(path: Path) -> list[tuple[int, str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []
    found = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in MCP_SERVER_CTORS
        ):
            continue
        if any(kw.arg == "name" for kw in node.keywords):
            continue
        pos = _name_position(node.func.id)
        if pos is not None and len(node.args) > pos:
            continue  # supplied positionally
        found.append((node.lineno, node.func.id))
    return found


def test_tracked_playgrounds_name_their_mcp_servers():
    offenders = []
    for path in _tracked_playground_files():
        for lineno, ctor in _unnamed_constructions(path):
            rel = path.relative_to(REPO_ROOT)
            offenders.append(f"{rel}:{lineno} {ctor}(...) has no name=")
    assert not offenders, "MCP servers constructed without name=:\n  " + "\n  ".join(offenders)


def test_the_audit_actually_scans_something():
    """Guard against the check passing because the file list came back empty --
    a silently-empty sweep reads exactly like a clean one."""
    files = _tracked_playground_files()
    assert len(files) > 10, files
    assert any(_has_ctor(p) for p in files), (
        "no MCP server constructions found at all; the AST matcher is likely broken"
    )


def _has_ctor(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return False
    return any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in MCP_SERVER_CTORS
        for n in ast.walk(tree)
    )


@pytest.mark.parametrize("ctor", sorted(MCP_SERVER_CTORS))
def test_ctor_names_still_exist_in_the_sdk(ctor):
    """If a transport is renamed, the audit above would quietly stop matching it."""
    import continuum.tools.mcp as mcp_mod

    assert hasattr(mcp_mod, ctor), f"{ctor} no longer exists; update MCP_SERVER_CTORS"
