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

from typing import TYPE_CHECKING, Any, Literal

from continuum.llm.untrusted_content import strip_hidden_chars
from continuum.logging import get_logger
from continuum.tools.mcp import _tool_digest

if TYPE_CHECKING:
    from mcp.types import Tool as MCPTool

    from continuum.tools.types import ToolFilterCallable, ToolFilterContext

logger = get_logger(__name__)

_DIGEST_PREVIEW = 12


def snapshot_tool_digests(server_name: str, tools: list[MCPTool]) -> dict[str, str]:
    """Map tool name to the digest of its description + schema.

    Uses the same :func:`~continuum.tools.mcp._tool_digest` as the live drift
    detector, so a pin captured here is directly comparable with what
    ``list_tools()`` records. If the two ever diverged, review and detection
    would disagree about what "unchanged" means.
    """
    del server_name  # accepted for symmetry with the CLI/pin-file shape
    return {tool.name: _tool_digest(tool) for tool in tools}


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
    approved: dict[str, str],
    *,
    on_unknown: Literal["block", "allow"] = "block",
) -> ToolFilterCallable:
    """Build a :data:`ToolFilter` that admits only tools matching ``approved``.

    Unlike the warn-and-re-pin tripwire in ``list_tools()``, this *blocks*: a
    tool whose description or schema drifted never reaches the model. Use it when
    you would rather an agent lose a tool than act on text you did not review.

    Args:
        approved: name -> digest, from :func:`snapshot_tool_digests` or a pin file.
        on_unknown: what to do with a tool absent from ``approved``. Defaults to
            ``"block"``: a tool that appeared after review was never approved, and
            the point of pinning is that only reviewed tools reach the model.

    Raises:
        ValueError: if ``approved`` is empty. That would drop every tool, which is
            almost certainly a caller passing an unpopulated pin file rather than
            intending a total block.
    """
    if not approved:
        raise ValueError(
            "create_tool_pinning_filter() received an empty approval map, which would "
            "drop every tool. Pass digests from snapshot_tool_digests(), or use "
            "create_static_tool_filter(allowed_tool_names=[]) if a total block is intended."
        )

    def _pinned(context: ToolFilterContext, tool: MCPTool) -> bool:
        expected = approved.get(tool.name)
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
