"""
Structured-output helpers — the single owner of "schema in, validated object out".

Used by both the non-streaming executor and the streaming runner so the two paths
cannot diverge. Three responsibilities:

  1. schema_prompt(schema)          -> a system instruction listing the exact JSON
                                       shape (the UNIVERSAL lever — works on every
                                       provider, including ones with no native
                                       response_format support).
  2. to_openai_response_format(...) -> an OpenAI json_schema response_format dict
                                       (the provider-specific enhancement, used on
                                       the streaming .create() path).
  3. coerce_and_validate(text, ...) -> parse model output text into a validated
                                       Pydantic instance, tolerating markdown
                                       fences / prose / single-key wrappers.

Design note: the model is steered toward the right shape PRIMARILY by the prompt
(schema_prompt) — native response_format is only a bonus where the provider
supports it. coerce_and_validate is the same logic that previously lived inline in
executor.py; it is intentionally tolerant because not every provider can be forced
to emit exactly-shaped JSON.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from continuum.llm.types import LLMResponse

# The synthetic tool used to enforce a schema on providers that have no
# response_format. A tool's parameter schema IS a JSON schema, and the API makes
# the model's arguments conform to it — so declaring one throwaway tool and
# forcing the model to call it turns "please emit this shape" into the only move
# available. Nothing is executed: the arguments are the answer.
#
# Shared by every provider that needs the trick, so the name they declare and
# the name they unwrap can never drift apart.
STRUCTURED_OUTPUT_TOOL = "emit_structured_output"
STRUCTURED_OUTPUT_TOOL_DESCRIPTION = (
    "Return the final answer. Call this exactly once, with arguments matching the schema."
)


def is_pydantic_schema(schema: object) -> bool:
    """True only if ``schema`` is an actual Pydantic model class.

    Guards the structured-output paths against a non-class ``output_schema``
    (e.g. a MagicMock in tests, or misuse) so we never call ``.__name__`` /
    ``model_validate`` on something that isn't a model.
    """
    return isinstance(schema, type) and issubclass(schema, BaseModel)


def schema_prompt(schema: type[BaseModel]) -> str:
    """Render a system instruction telling the model the exact JSON shape to emit.

    This is the cross-provider floor: every model reads the prompt, even those that
    ignore (or don't support) a native response_format. We list the field names and
    types explicitly so the model uses *our* keys rather than inventing its own.
    """
    fields = []
    for name, field in schema.model_fields.items():
        annotation = getattr(field.annotation, "__name__", str(field.annotation))
        required = "required" if field.is_required() else "optional"
        desc = f" — {field.description}" if field.description else ""
        fields.append(f'  "{name}": {annotation}  ({required}){desc}')
    field_block = "\n".join(fields)

    return (
        "You MUST respond with ONLY a single JSON object that exactly matches this "
        "schema — no markdown fences, no prose, no extra keys:\n"
        f"{field_block}\n"
        f"Use exactly these field names: {', '.join(schema.model_fields)}."
    )


def to_openai_response_format(schema: type[BaseModel]) -> dict[str, Any]:
    """Build an OpenAI-style ``json_schema`` response_format dict from a model.

    Used on the streaming ``.create()`` path (which cannot take a Pydantic class
    the way ``.parse()`` can). Not marked ``strict`` — strict mode requires schema
    transforms (all-required, additionalProperties:false) that many real models
    don't satisfy; we rely on coerce_and_validate + the prompt floor for shape.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "schema": schema.model_json_schema(),
        },
    }


def openai_schema_tool(schema: dict[str, Any]) -> dict[str, Any]:
    """OpenAI-wire ``tools``/``tool_choice`` pair that forces ``schema``.

    The second way to ask for a shape (see STRUCTURED_OUTPUT_TOOL). Used for
    upstreams that speak the OpenAI protocol but drop ``response_format`` on the
    way to their real backend — the Smart Gateway's Bedrock path, which
    translates ``tools``/``tool_choice`` into Converse ``toolConfig`` but has no
    entry for ``response_format`` at all.
    """
    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": STRUCTURED_OUTPUT_TOOL,
                    "description": STRUCTURED_OUTPUT_TOOL_DESCRIPTION,
                    "parameters": schema,
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": STRUCTURED_OUTPUT_TOOL}},
    }


def forces_structured_tool(kwargs: dict[str, Any]) -> bool:
    """True if this request forced the synthetic schema tool.

    Reads the built request rather than a flag carried alongside it, so the
    response handling cannot drift out of step with what was actually sent.
    Recognises both spellings: Anthropic's ``{"type": "tool", "name": …}`` and
    the OpenAI wire's ``{"type": "function", "function": {"name": …}}``.
    """
    choice = kwargs.get("tool_choice")
    if not isinstance(choice, dict):
        return False
    function = choice.get("function")
    name = choice.get("name") or (function.get("name") if isinstance(function, dict) else None)
    return bool(name == STRUCTURED_OUTPUT_TOOL)


def unwrap_structured_tool_call(response: LLMResponse) -> LLMResponse:
    """Present a forced schema answer as ordinary content.

    The answer arrives as the synthetic tool's arguments. Left as a tool call,
    the executor would try to *execute* a tool that does not exist; moving the
    arguments into ``content`` puts the JSON exactly where every caller —
    coerce_and_validate, AgentResponse.content — already looks.

    Matched by tool name rather than "a tool was called", so a genuine tool call
    is never swallowed.
    """
    for call in response.tool_calls or []:
        if call.function.name == STRUCTURED_OUTPUT_TOOL:
            return response.model_copy(
                update={
                    "content": call.function.arguments,
                    "tool_calls": None,
                    # The turn stopped to "call a tool", but no work is owed:
                    # as far as the run is concerned this is a final answer.
                    "finish_reason": "stop",
                }
            )
    return response


def schema_from_response_format(response_format: Any) -> dict[str, Any] | None:
    """Pull the bare JSON Schema out of an OpenAI-style ``response_format``.

    ``response_format`` is Continuum's neutral carrier for "the answer must have
    this shape", but its spelling is OpenAI's. Providers that enforce schemas a
    different way (Anthropic's forced tool use, Gemini's ``responseSchema``) need
    the schema itself, not the envelope. Returns None when the format names no
    schema — ``{"type": "json_object"}`` asks for *some* JSON of *any* shape, so
    there is nothing to enforce against.
    """
    if not isinstance(response_format, dict):
        return None
    if response_format.get("type") != "json_schema":
        return None
    block = response_format.get("json_schema")
    if not isinstance(block, dict):
        return None
    schema = block.get("schema")
    return schema if isinstance(schema, dict) else None


def looks_like_json(content: str) -> bool:
    """True if ``content`` parses as JSON once fences and prose are stripped.

    The JSON-mode warnings used to test the raw string, so a perfectly good
    ```json fenced block — which ``coerce_and_validate`` recovers without
    trouble — warned on every single call. Judge what the parser will actually
    see, not what the model happened to wrap it in.
    """
    if not content:
        return False
    try:
        json.loads(_strip_to_json(content))
    except (json.JSONDecodeError, ValueError):
        return False
    return True


def _strip_to_json(content: str) -> str:
    """Strip markdown fences and, if needed, extract an embedded JSON object/array."""
    s = content.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-z]*\n?", "", s)
        s = s.rstrip("`").rstrip()

    is_json_like = (s.startswith("{") and s.endswith("}")) or (
        s.startswith("[") and s.endswith("]")
    )
    if not is_json_like:
        # Try to dig a JSON object/array out of prose or markdown.
        match = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", s)
        if match:
            s = match.group(0)
    return s


def _unwrap_schema_key(parsed: Any, schema: type[BaseModel]) -> Any:
    """Unwrap ``{"SchemaName": {...}}`` if the model wrapped the payload by name."""
    if isinstance(parsed, dict) and len(parsed) == 1:
        only_key = next(iter(parsed))
        if only_key == schema.__name__:
            inner = parsed[only_key]
            return {"leads": inner} if isinstance(inner, list) else inner
    return parsed


def coerce_and_validate(
    content: str | None, schema: type[BaseModel]
) -> tuple[BaseModel | None, str | None]:
    """Parse ``content`` into a validated instance of ``schema``.

    Returns ``(instance, None)`` on success, or ``(None, error_message)`` on
    failure (invalid JSON or schema mismatch). Never raises — the caller decides
    whether to surface the error softly or raise.
    """
    if not content:
        return None, "empty content"

    try:
        stripped = _strip_to_json(content)
        parsed = json.loads(stripped)
    except json.JSONDecodeError as e:
        return None, f"response is not valid JSON: {e}"

    parsed = _unwrap_schema_key(parsed, schema)

    try:
        return schema.model_validate(parsed), None
    except ValidationError as e:
        return None, f"response did not match schema {schema.__name__}: {e}"
