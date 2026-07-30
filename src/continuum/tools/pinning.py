"""Human review and enforcement for MCP tool catalogues (security finding F3).

The drift detection in ``list_tools()`` answers *"did this change since I last
looked?"*. It cannot answer *"was it safe the first time"* -- pin a
born-malicious server and you have pinned the poison. Nothing in a framework can
close that gap: whether a third-party server is trustworthy is a judgement, and
the MCP specification assigns it to the host application.

What a framework *can* do is make the judgement practical. Before this module
there was no way to see what a server ships -- the CLI offered only
``up``/``down``/``status``/``logs``/``config-path``, so reviewing a catalogue
meant hand-writing an async script. "Read the descriptions before you trust a
server" was correct advice that nobody could act on.

So:

  * :func:`format_tool_catalog` renders a catalogue for a person to read;
  * :func:`snapshot_tool_digests` captures what was reviewed;
  * :func:`create_tool_pinning_filter` turns that snapshot into a
    :data:`~continuum.tools.types.ToolFilter` that drops anything which no longer
    matches.

Note the intended order: review first, pin second. A helper that produced pins
without showing you the contents would make *pin-without-reading* the one-liner
and *read-then-pin* the chore -- optimising the dangerous path. That is why the
CLI command prints the catalogue and treats the pin file as a byproduct.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from continuum.llm.untrusted_content import strip_hidden_chars
from continuum.logging import get_logger
from continuum.tools.mcp import _tool_digest
from continuum.tools.util import build_namespaced_tool_name

if TYPE_CHECKING:
    from mcp.types import Tool as MCPTool

    from continuum.tools.types import ToolFilterCallable, ToolFilterContext

logger = get_logger(__name__)

_DIGEST_PREVIEW = 12

PIN_FORMAT_VERSION = 1
"""On-disk schema version for pin files.

Every lockfile carries one; this one didn't, and the cost was a compatibility
branch that had to *infer* whether a bare digest string covered the raw or the
cleaned catalogue -- a question the file could not answer. A version field turns
"guess the shape" into "refuse to read what you don't understand".
"""

PinEntry = dict[str, Any]
ServerPins = dict[str, PinEntry]


def snapshot_tool_digests(server_name: str, tools: list[MCPTool]) -> ServerPins:
    """Map tool name to its pin entry: two digests plus the reviewed text.

    Two digests because the two consumers ask different questions, and a single
    digest silently broke one of them:

    ``raw``
        Over the bytes exactly as the server sent them. What the drift tripwire
        in ``list_tools()`` compares, so that adding or removing invisible
        characters cannot slip past unreported.

    ``effective``
        Over the catalogue after :func:`~continuum.tools.mcp._clean_tool` has
        stripped invisible characters -- i.e. the text the model will actually
        receive. What the pinning gate compares, because ``list_tools()`` cleans
        tools *before* handing them to a ``tool_filter``.

    They are identical for ordinary text; only a description carrying invisible
    characters makes them differ. Comparing a cleaned tool against a raw pin
    meant such a tool could never match, so the gate dropped it forever while
    reporting that it "no longer matches" -- which was never achievable.

    ``description`` and ``inputSchema`` are stored alongside because a digest
    records *that* something changed and can never record *what*. Every
    human-facing step -- reviewing a diff, deciding whether to approve -- needs
    the previous text, and ``- "82f31…" + "9db49…"`` is not reviewable.
    """
    del server_name  # accepted for symmetry with the CLI/pin-file shape
    from continuum.tools.mcp import _clean_tool

    return {
        tool.name: {
            "raw": _tool_digest(tool),
            "effective": _tool_digest(_clean_tool(tool)),
            "description": tool.description or "",
            "inputSchema": tool.inputSchema or {},
        }
        for tool in tools
    }


def load_pins(path: str | Path) -> dict[str, ServerPins]:
    """Read a pin file, returning ``server name -> tool name -> entry``.

    Best-effort by design: a missing, unreadable, corrupt, or
    future-versioned file degrades to "no baseline" rather than taking an agent
    down. Trust *bookkeeping* is advisory; the enforcement decision built on top
    of it is where failures should be loud.

    A version this code does not recognise is refused rather than parsed
    optimistically -- a newer writer may mean something different by the same
    keys, and quietly misreading an approval is worse than having none.
    """
    file = Path(path)
    try:
        raw = json.loads(file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        logger.warning(f"Ignoring unreadable MCP tool-pin file {file}: {e}")
        return {}

    if not isinstance(raw, dict):
        logger.warning(f"Ignoring malformed MCP tool-pin file {file}: expected a JSON object.")
        return {}

    version = raw.get("version")
    if version != PIN_FORMAT_VERSION:
        logger.warning(
            f"Ignoring MCP tool-pin file {file}: unsupported format version {version!r} "
            f"(this build reads version {PIN_FORMAT_VERSION}). Re-create it with "
            f"`continuum mcp inspect URL --name SERVER --approve {file}`."
        )
        return {}

    servers = raw.get("servers")
    if not isinstance(servers, dict):
        logger.warning(f"Ignoring MCP tool-pin file {file}: no 'servers' object.")
        return {}

    return {
        name: {tool: entry for tool, entry in pins.items() if isinstance(entry, dict)}
        for name, pins in servers.items()
        if isinstance(pins, dict)
    }


def save_pins(path: str | Path, servers: dict[str, ServerPins]) -> None:
    """Write ``servers`` to ``path`` in the current pin-file format.

    Whole-file write: callers pass the complete mapping, so merging is an
    explicit step they perform rather than something this function guesses at.
    Replacing a server entry wholesale is exactly the bug that made approval
    all-or-nothing -- one drifted tool forced you to either bless every other
    change or lose the tool.

    Sorted keys because the file is meant to be diffed; a reordering diff is a
    diff nobody reads.
    """
    file = Path(path)
    payload = {"version": PIN_FORMAT_VERSION, "servers": servers}
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _format_schema_descriptions(node: Any, path: str = "") -> list[str]:
    """Collect ``path: description`` lines from a JSON Schema.

    Parameter descriptions are a second injection surface -- the F3 proof of
    concept smuggles via "...include its contents in the notes field" -- so a
    review that showed only the tool-level description would miss it.
    """
    lines: list[str] = []
    if isinstance(node, dict):
        if isinstance(node.get("description"), str) and path:
            lines.append(f"{path}: {node['description']}")
        for key, value in node.items():
            if key == "description":
                continue
            child = f"{path}.{key}" if path and key != "properties" else path or key
            lines.extend(_format_schema_descriptions(value, child if key != "properties" else path))
    elif isinstance(node, list):
        for item in node:
            lines.extend(_format_schema_descriptions(item, path))
    return lines


def format_tool_catalog(server_name: str, tools: list[MCPTool]) -> str:
    """Render a catalogue for human review.

    Descriptions are shown **in full and unabridged**: truncating would defeat
    the purpose, since a payload appended to an otherwise ordinary description
    hides in the tail.

    Invisible characters are *reported*, not silently removed. The live path
    strips them (see ``_clean_tool``), but a reviewer must be told the server
    sent hidden text -- that is the single strongest signal a server is hostile,
    and quietly cleaning it away would conceal exactly what review is for.
    """
    if not tools:
        return f"Server '{server_name}': no tools reported."

    out: list[str] = [
        f"Server '{server_name}' — {len(tools)} tool(s)",
        "",
        "Read every description below before trusting this server. Text here reaches",
        "the model's prompt verbatim and instructs it; a poisoned tool keeps an",
        "innocent-looking name.",
        "",
    ]

    for tool in tools:
        digest = _tool_digest(tool)
        description = tool.description or ""
        cleaned = strip_hidden_chars(description)

        out.append(f"{'─' * 72}")
        out.append(f"{tool.name}   [digest {digest[:_DIGEST_PREVIEW]}]")
        # The raw name above is what the server calls it; this is what the model
        # sees and what PolicyStore / always_promote match. With an auto-derived
        # server name the prefix is sanitized and may be hash-truncated, so it is
        # not something a reader can work out -- print it rather than expect them
        # to guess, or they write tool:delete_user, match nothing, and a deny
        # rule silently stops denying.
        out.append(f"  policy resource:  tool:{build_namespaced_tool_name(server_name, tool.name)}")
        out.append("")
        out.append(f"  {description or '(no description)'}")

        if cleaned != description:
            removed = len(description) - len(cleaned)
            out.append("")
            out.append(
                f"  *** WARNING: {removed} hidden/invisible character(s) in this description. ***"
            )
            out.append("  These are readable by the model but not by you. Treat this server")
            out.append("  as hostile unless you can explain them.")
            out.append(f"  Visible text only: {cleaned!r}")

        param_lines = _format_schema_descriptions(tool.inputSchema or {})
        if param_lines:
            out.append("")
            out.append("  Parameters:")
            out.extend(f"    {line}" for line in param_lines)

    out.append(f"{'─' * 72}")
    return "\n".join(out)


def create_tool_pinning_filter(
    approved: dict[str, dict[str, str]],
    *,
    on_unknown: Literal["block", "allow"] = "block",
) -> ToolFilterCallable:
    """Build a :data:`ToolFilter` that admits only tools matching ``approved``.

    Unlike the warn-and-re-pin tripwire in ``list_tools()``, this *blocks*: a
    tool whose description or schema drifted never reaches the model. Use it when
    you would rather an agent lose a tool than act on text you did not review.

    Compares the ``effective`` digest -- the catalogue *after* invisible
    characters are stripped, which is what ``list_tools()`` hands a
    ``tool_filter`` and what the model will read. So a description whose only
    change is hidden characters still passes here: those never reach the model,
    and reporting them is the drift tripwire's job, not this gate's.

    Args:
        approved: ``name -> {"raw": ..., "effective": ...}``, from
            :func:`snapshot_tool_digests` or a pin file written by
            ``continuum mcp inspect --write-pins``.
        on_unknown: what to do with a tool absent from ``approved``. Defaults to
            ``"block"``: a tool that appeared after review was never approved, and
            the point of pinning is that only reviewed tools reach the model.

    Raises:
        ValueError: if ``approved`` is empty (that would drop every tool, almost
            certainly an unpopulated pin file rather than an intended total
            block), or if it holds bare digest strings from the single-digest pin
            format -- those carry no indication of which digest space they are in,
            and guessing would either drop every tool or admit a changed one.
    """
    if not approved:
        raise ValueError(
            "create_tool_pinning_filter() received an empty approval map, which would "
            "drop every tool. Pass digests from snapshot_tool_digests(), or use "
            "create_static_tool_filter(allowed_tool_names=[]) if a total block is intended."
        )

    effective: dict[str, str] = {}
    for name, entry in approved.items():
        if isinstance(entry, str):
            raise ValueError(
                f"Pin for {name!r} is a bare digest string from the old single-digest "
                f"format, which does not record whether it covers the raw or the "
                f"cleaned catalogue. Re-pin: "
                f"`continuum mcp inspect URL --name SERVER --write-pins PATH`."
            )
        if not isinstance(entry, dict) or "effective" not in entry:
            raise ValueError(
                f"Pin for {name!r} is malformed (expected a mapping with an "
                f"'effective' key, got {entry!r}). Re-pin with "
                f"`continuum mcp inspect --write-pins PATH`."
            )
        effective[name] = entry["effective"]

    def _pinned(context: ToolFilterContext, tool: MCPTool) -> bool:
        expected = effective.get(tool.name)
        if expected is None:
            if on_unknown == "allow":
                return True
            logger.warning(
                f"Tool '{tool.name}' on server '{context.server_name}' is not in the "
                f"approved set — dropping. Re-run `continuum mcp inspect` to review "
                f"and re-pin if this addition is expected."
            )
            return False
        if _tool_digest(tool) != expected:
            logger.warning(
                f"Tool '{tool.name}' on server '{context.server_name}' no longer matches "
                f"its approved description/schema — dropping. If you did not change this "
                f"server, treat it as untrusted."
            )
            return False
        return True

    return _pinned
