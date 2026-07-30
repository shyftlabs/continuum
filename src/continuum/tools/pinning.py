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
  * :func:`load_pins` / :func:`save_pins` persist it in a versioned file that
    only a human command writes;
  * :func:`diff_catalogs` and :func:`format_catalog_diff` show what changed
    since;
  * :func:`approve_tools` promotes reviewed entries, one tool at a time.

Enforcement is not here: it belongs to the server, via
:class:`~continuum.tools.types.ToolTrustConfig`'s ``on_unreviewed`` and
``on_drift``. A standalone filter used to do that job, but it had to be paired
with no pin path or the drift tripwire would rewrite the file the filter read --
silently promoting an attacker's catalogue to "approved" one restart later.
Folding it into the server removes the way to misconfigure it.

Note the intended order: review first, pin second. A helper that produced pins
without showing you the contents would make *pin-without-reading* the one-liner
and *read-then-pin* the chore -- optimising the dangerous path. That is why the
CLI command prints the catalogue and treats the pin file as a byproduct.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from continuum.llm.untrusted_content import strip_hidden_chars
from continuum.logging import get_logger
from continuum.tools.mcp import _tool_digest
from continuum.tools.util import build_namespaced_tool_name

if TYPE_CHECKING:
    from mcp.types import Tool as MCPTool


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


@dataclass
class ToolDiff:
    """One tool's difference between the approved catalogue and a later one.

    Carries the full entries, not just a verdict, because the point of the diff
    is that a person reads the text and decides. ``approved`` is None for a tool
    that appeared after review; ``current`` is None for one that vanished.
    """

    name: str
    status: Literal["changed", "added", "removed"]
    approved: PinEntry | None
    current: PinEntry | None

    @property
    def hidden_char_delta(self) -> int:
        """How many invisible characters this change adds.

        Reported rather than silently cleaned: the live path strips them, but a
        reviewer must be told the server sent hidden text. It is the single
        strongest signal a server is hostile, and hiding it defeats the review.
        """
        after = self.current.get("description", "") if self.current else ""
        before = self.approved.get("description", "") if self.approved else ""
        return (len(after) - len(strip_hidden_chars(after))) - (
            len(before) - len(strip_hidden_chars(before))
        )


def diff_catalogs(approved: ServerPins, current: ServerPins) -> list[ToolDiff]:
    """Compare an approved catalogue against a later one.

    Returns only the differences, sorted by tool name: a review should list
    what needs deciding, not restate the whole catalogue. Sorted because
    unstable ordering makes two runs of the same command look different.

    Comparison is over the ``raw`` digest -- the bytes as the server sent them
    -- so that adding or removing invisible characters cannot slip past.
    """
    names = sorted(set(approved) | set(current))
    diffs: list[ToolDiff] = []
    for name in names:
        before, after = approved.get(name), current.get(name)
        if before is None:
            diffs.append(ToolDiff(name, "added", None, after))
        elif after is None:
            diffs.append(ToolDiff(name, "removed", before, None))
        elif before.get("raw") != after.get("raw"):
            diffs.append(ToolDiff(name, "changed", before, after))
    return diffs


def _gutter(marker: str, description: str | None) -> list[str]:
    """Render a description with every line inside the +/- gutter.

    Descriptions are multi-line, and a payload is typically appended on a new
    one. Marking only the first line would leave the injected text flush-left,
    where it reads as commentary about the diff rather than as part of the
    description being added -- which is precisely the misreading that helps an
    attacker.
    """
    text = description or "(no description)"
    return [f"  {marker} {line}" for line in text.splitlines() or [""]]


def format_catalog_diff(server_name: str, diffs: list[ToolDiff]) -> str:
    """Render differences for a person to read and act on."""
    if not diffs:
        return f"Server '{server_name}': no differences from the approved catalogue."

    out: list[str] = [
        f"Server '{server_name}' — {len(diffs)} unreviewed difference(s)",
        "",
    ]
    for diff in diffs:
        out.append("─" * 72)
        header = f"{diff.name}   [{diff.status}]"
        if diff.hidden_char_delta > 0:
            header += f"   *** {diff.hidden_char_delta} hidden character(s) added ***"
        out.append(header)
        out.append("")
        if diff.approved is not None:
            out.extend(_gutter("-", diff.approved.get("description")))
        if diff.current is not None:
            out.extend(_gutter("+", diff.current.get("description")))
        if diff.approved is not None and diff.current is not None:
            before = diff.approved.get("inputSchema")
            after = diff.current.get("inputSchema")
            if before != after:
                out.append("")
                out.append(f"  - schema: {json.dumps(before, sort_keys=True)}")
                out.append(f"  + schema: {json.dumps(after, sort_keys=True)}")

    out.append("─" * 72)
    out.append("")
    # A report of a problem carries its own remedy: without the command, the
    # reader knows something is wrong and not what to do about it.
    out.append("Approve the changes you have read and accept:")
    out.append(f"  continuum mcp approve {server_name} --tool NAME")
    out.append(f"  continuum mcp approve {server_name} --all")
    return "\n".join(out)


def approve_tools(
    pin_path: str | Path,
    server_name: str,
    current: ServerPins,
    *,
    tools: list[str] | None = None,
) -> list[str]:
    """Merge selected entries of ``current`` into the approved catalogue.

    Merge, never replace. Replacing a server's whole entry is what made
    approval all-or-nothing: with one benign typo fix and one injection, you had
    to either bless the injection to recover the typo fix or lose the unrelated
    tool for good. Enforcement was already per-tool; this makes approval match.

    Args:
        pin_path: the approved catalogue. Created if absent.
        server_name: which server's entry to update; others are left alone.
        current: the catalogue being approved *from* -- usually the runtime's
            last-seen record, i.e. the text that was just shown in a diff.
        tools: names to approve. ``None`` approves the whole of ``current``.
            A name absent from ``current`` but present in the approved
            catalogue is a removal, and approving it drops the entry.

    Returns:
        The names acted on, sorted.

    Raises:
        ValueError: on an empty selection, or a name in neither catalogue --
            approving nothing while reporting success would let a typo'd tool
            name read as "approved".
    """
    existing = load_pins(pin_path)
    approved = dict(existing.get(server_name, {}))

    if tools is None:
        selected = sorted(set(current) | set(approved))
    else:
        selected = sorted(set(tools))
        if not selected:
            raise ValueError(
                "approve_tools() received an empty tool selection. Pass tools=None to "
                "approve the whole catalogue."
            )
        unknown = [t for t in selected if t not in current and t not in approved]
        if unknown:
            raise ValueError(
                f"No such tool(s) on server {server_name!r}: {unknown}. "
                f"Check for a typo -- approving a name that does not exist would "
                f"report success and change nothing. "
                f"Known: {sorted(set(current) | set(approved))}"
            )

    for name in selected:
        if name in current:
            approved[name] = current[name]
        else:
            approved.pop(name, None)

    existing[server_name] = approved
    save_pins(pin_path, existing)
    return selected


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


