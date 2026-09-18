"""
Schema normalization utilities for LLM-agnostic MCP tool integration.

This module provides functions to transform MCP tool schemas into a format
that works reliably across all LLM providers (OpenAI, Gemini, Anthropic, etc.).

The normalization ensures:
1. All array types have 'items' field (required by OpenAI, Gemini)
2. All object types have 'properties' field (required by OpenAI)
3. Strict mode support with 'required' and 'additionalProperties'
4. Recursive handling of nested schemas, anyOf, oneOf, allOf
"""

from copy import deepcopy
from typing import Any

from continuum.logging import get_logger

logger = get_logger(__name__)

# JSON Schema keywords that contain nested schemas
SCHEMA_KEYWORDS_WITH_SUBSCHEMAS = frozenset(
    ["anyOf", "oneOf", "allOf", "not", "if", "then", "else"]
)

# Default schema for arrays without items (must be valid for OpenAI/Gemini)
# An empty {} schema is rejected by LLM providers as "missing items"
# Using anyOf with all types preserves the "any type" semantics of empty schema
# This ensures LLM can generate any type of value, matching the original intent
DEFAULT_ARRAY_ITEMS: dict[str, Any] = {
    "anyOf": [
        {"type": "string"},
        {"type": "number"},
        {"type": "integer"},
        {"type": "boolean"},
        {"type": "object"},
        {"type": "array", "items": {"type": "string"}},  # Nested array needs items
        {"type": "null"},
    ]
}

# Default type when type is missing
DEFAULT_TYPE = "object"


def normalize_schema_for_llm(
    schema: dict[str, Any],
    strict: bool = False,
    _path: str = "root",
) -> dict[str, Any]:
    """
    Normalize an MCP tool schema for LLM provider compatibility.

    This function transforms MCP schemas to work reliably across all major
    LLM providers (OpenAI, Gemini, Anthropic, Mistral, etc.).

    Transformations applied:
    1. Arrays without 'items' get 'items': {} added
    2. Objects without 'properties' get 'properties': {} added
    3. Missing 'type' is inferred or defaulted to 'object'
    4. Nested schemas (in anyOf, oneOf, etc.) are recursively normalized
    5. If strict=True: adds 'required' and 'additionalProperties: false'

    Args:
        schema: The JSON schema to normalize (from MCP tool inputSchema).
        strict: If True, enable strict mode (all props required, no additional).
        _path: Internal parameter for debug logging (tracks schema path).

    Returns:
        A new normalized schema dict (original is not modified).

    Example:
        >>> schema = {"type": "object", "properties": {"items": {"type": "array"}}}
        >>> normalized = normalize_schema_for_llm(schema, strict=True)
        >>> # Result: items array now has 'items': {}, object has 'required' and 'additionalProperties'
    """
    if not isinstance(schema, dict):
        return schema

    # Deep copy to avoid mutating original
    result = deepcopy(schema)

    # Normalize this schema node
    result = _normalize_schema_node(result, strict, _path)

    return result


def _normalize_schema_node(
    schema: dict[str, Any],
    strict: bool,
    path: str,
) -> dict[str, Any]:
    """
    Normalize a single schema node and recursively process children.

    Args:
        schema: The schema node to normalize.
        strict: Whether to apply strict mode transformations.
        path: Current path in schema for logging.

    Returns:
        Normalized schema node.
    """
    schema_type = schema.get("type")

    # Handle missing type - infer from context
    if schema_type is None:
        schema_type = _infer_type(schema)
        if schema_type:
            schema["type"] = schema_type
            logger.debug(f"Inferred type '{schema_type}' at {path}")

    # Normalize based on type
    if schema_type == "array":
        schema = _normalize_array_schema(schema, strict, path)
    elif schema_type == "object":
        schema = _normalize_object_schema(schema, strict, path)

    # Handle composite schemas (anyOf, oneOf, allOf, etc.)
    for keyword in SCHEMA_KEYWORDS_WITH_SUBSCHEMAS:
        if keyword in schema:
            value = schema[keyword]
            if isinstance(value, list):
                # anyOf, oneOf, allOf contain arrays of schemas
                schema[keyword] = [
                    _normalize_schema_node(
                        sub_schema if isinstance(sub_schema, dict) else sub_schema,
                        strict,
                        f"{path}.{keyword}[{i}]",
                    )
                    if isinstance(sub_schema, dict)
                    else sub_schema
                    for i, sub_schema in enumerate(value)
                ]
            elif isinstance(value, dict):
                # 'not', 'if', 'then', 'else' contain single schema
                schema[keyword] = _normalize_schema_node(value, strict, f"{path}.{keyword}")

    return schema


def _normalize_array_schema(
    schema: dict[str, Any],
    strict: bool,
    path: str,
) -> dict[str, Any]:
    """
    Normalize an array type schema.

    Ensures 'items' field exists and is valid (required by OpenAI, Gemini).
    An empty items schema {} is rejected by LLM providers.

    Args:
        schema: Array schema to normalize.
        strict: Whether to apply strict mode.
        path: Current path for logging.

    Returns:
        Normalized array schema.
    """
    # Ensure 'items' exists - this is the critical fix for LLM compatibility
    if "items" not in schema:
        schema["items"] = DEFAULT_ARRAY_ITEMS.copy()
        logger.debug(f"Added missing 'items' to array at {path}")
    # Handle empty items schema {} - LLM providers reject this as invalid
    elif isinstance(schema.get("items"), dict) and not schema["items"]:
        schema["items"] = DEFAULT_ARRAY_ITEMS.copy()
        logger.debug(f"Replaced empty 'items' schema with default at {path}")

    # Recursively normalize items schema
    items = schema.get("items")
    if isinstance(items, dict):
        schema["items"] = _normalize_schema_node(items, strict, f"{path}.items")

    # Handle tuple validation (items as array)
    if isinstance(items, list):
        schema["items"] = [
            _normalize_schema_node(item, strict, f"{path}.items[{i}]")
            if isinstance(item, dict)
            else item
            for i, item in enumerate(items)
        ]

    # Normalize additionalItems if present
    if "additionalItems" in schema and isinstance(schema["additionalItems"], dict):
        schema["additionalItems"] = _normalize_schema_node(
            schema["additionalItems"], strict, f"{path}.additionalItems"
        )

    # Normalize contains if present
    if "contains" in schema and isinstance(schema["contains"], dict):
        schema["contains"] = _normalize_schema_node(schema["contains"], strict, f"{path}.contains")

    return schema


def _normalize_object_schema(
    schema: dict[str, Any],
    strict: bool,
    path: str,
) -> dict[str, Any]:
    """
    Normalize an object type schema.

    Ensures 'properties' field exists (required by OpenAI).
    In strict mode, adds 'required' and 'additionalProperties: false'.

    Args:
        schema: Object schema to normalize.
        strict: Whether to apply strict mode.
        path: Current path for logging.

    Returns:
        Normalized object schema.
    """
    # Ensure 'properties' exists
    if "properties" not in schema:
        schema["properties"] = {}
        logger.debug(f"Added missing 'properties' to object at {path}")

    # Recursively normalize each property
    properties = schema.get("properties", {})
    if isinstance(properties, dict):
        for prop_name, prop_schema in properties.items():
            if isinstance(prop_schema, dict):
                properties[prop_name] = _normalize_schema_node(
                    prop_schema, strict, f"{path}.properties.{prop_name}"
                )

    # Normalize additionalProperties if it's a schema
    if "additionalProperties" in schema and isinstance(schema["additionalProperties"], dict):
        schema["additionalProperties"] = _normalize_schema_node(
            schema["additionalProperties"], strict, f"{path}.additionalProperties"
        )

    # Normalize patternProperties if present
    if "patternProperties" in schema and isinstance(schema["patternProperties"], dict):
        for pattern, pattern_schema in schema["patternProperties"].items():
            if isinstance(pattern_schema, dict):
                schema["patternProperties"][pattern] = _normalize_schema_node(
                    pattern_schema, strict, f"{path}.patternProperties.{pattern}"
                )

    # Apply strict mode transformations
    if strict:
        schema = _apply_strict_mode(schema, path)

    return schema


def _apply_strict_mode(schema: dict[str, Any], path: str) -> dict[str, Any]:
    """
    Apply strict mode transformations to an object schema.

    Strict mode ensures:
    1. All properties are required
    2. No additional properties allowed

    This enables OpenAI's strict mode which guarantees LLM outputs
    match the schema exactly.

    Args:
        schema: Object schema to make strict.
        path: Current path for logging.

    Returns:
        Schema with strict mode applied.
    """
    properties = schema.get("properties", {})

    # Add all properties to 'required' if not already set
    if "required" not in schema and properties:
        schema["required"] = list(properties.keys())
        logger.debug(f"Added 'required' with all properties at {path}")

    # Disallow additional properties
    if "additionalProperties" not in schema:
        schema["additionalProperties"] = False
        logger.debug(f"Set 'additionalProperties: false' at {path}")

    return schema


def _infer_type(schema: dict[str, Any]) -> str | None:
    """
    Infer the type of a schema from its structure.

    Args:
        schema: Schema to infer type for.

    Returns:
        Inferred type string or None if cannot be determined.
    """
    # If has 'properties', 'additionalProperties', 'patternProperties' -> object
    if any(key in schema for key in ["properties", "additionalProperties", "patternProperties"]):
        return "object"

    # If has 'items', 'additionalItems', 'contains' -> array
    if any(key in schema for key in ["items", "additionalItems", "contains"]):
        return "array"

    # If has 'enum' but no type, don't infer (could be any type)
    if "enum" in schema:
        return None

    # If has anyOf/oneOf/allOf, don't infer type
    if any(key in schema for key in SCHEMA_KEYWORDS_WITH_SUBSCHEMAS):
        return None

    # Default: if schema has keys but no type, default to object
    # This handles cases like {"description": "...", "default": ...}
    if schema and "type" not in schema:
        return DEFAULT_TYPE

    return None


def ensure_strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Convenience function to normalize schema with strict mode enabled.

    This is equivalent to normalize_schema_for_llm(schema, strict=True).

    Use this when you want to enable OpenAI's strict mode for guaranteed
    schema compliance in LLM outputs.

    Args:
        schema: The JSON schema to normalize.

    Returns:
        Normalized schema with strict mode applied.
    """
    return normalize_schema_for_llm(schema, strict=True)


# =============================================================================
# Argument validation (security finding F5)
# =============================================================================

# JSON Schema type name -> predicate over a decoded JSON value.
#
# ``bool`` is a subclass of ``int`` in Python but a separate type in JSON
# Schema, so "integer"/"number" must exclude it explicitly -- otherwise `True`
# satisfies a parameter declared ``{"type": "integer"}`` and arrives in the tool
# body as 1. "number" accepts ``int`` because JSON has no int/float split: 3 is
# a valid number and rejecting it would fail every correctly-behaved provider.
_TYPE_PREDICATES: dict[str, Any] = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
    "null": lambda v: v is None,
}


def _type_matches(declared: Any, value: Any) -> bool:
    """Whether ``value`` satisfies a schema's ``type``, which may be a list.

    An unrecognised type name passes: this validator's job is to enforce what it
    understands, not to reject schemas written against a wider vocabulary.
    """
    names = declared if isinstance(declared, list) else [declared]
    checked = [_TYPE_PREDICATES[n] for n in names if n in _TYPE_PREDICATES]
    if not checked:
        return True
    return any(pred(value) for pred in checked)


def validate_arguments_against_schema(
    schema: dict[str, Any] | None,
    arguments: dict[str, Any],
    *,
    tool_name: str,
) -> None:
    """Check ``arguments`` against a tool's declared ``inputSchema``.

    Returns ``None`` on success and raises ``ToolArgumentError`` naming the
    offending parameter on failure. Reporting the parameter matters more than
    usual here: the reader is the model, on its next turn, deciding what to send
    instead.

    Enforced: required keys present, unknown keys absent, top-level ``type`` and
    ``enum`` per property.

    Deliberately *not* enforced:

    - **Nested structure.** A property declared ``{"type": "array"}`` is checked
      to be a list and no further. This mirrors what the generator can express:
      ``_type_to_schema`` emits container types without ``items``, so validating
      deeper would only ever fire on hand-written schemas -- an inconsistency
      that reads as a bug. Top-level shape is where the LLM-boundary confusions
      actually land.
    - **Anything under an empty ``{}`` property schema.** An open schema declares
      no constraint, and inventing one would reject arguments the model was
      correctly told were acceptable.
    - **Unknown keys, when the schema does not list ``properties``** (a bare
      ``{"type": "object"}``) or sets ``additionalProperties: True``. Both are
      explicit statements that the argument set is open.

    What this cannot do is worth stating plainly, because the gap is easy to
    misread as covered: a *validly typed* argument is still arbitrary attacker
    text. ``query="'; DROP TABLE users; --"`` satisfies ``{"type": "string"}``
    and always will. No schema rejects it. In-process tools run with the calling
    process's full authority, so a tool body must treat its own arguments as
    hostile input -- parameterise the query, resolve the path, check the scope.
    This function narrows the shape of what arrives; it does not sanitise it.

    Args:
        schema: The tool's declared ``inputSchema``. ``None`` or a non-dict
            schema declares nothing and so validates everything.
        arguments: Decoded arguments as they would be passed to the tool.
        tool_name: Named in the error, for the model's benefit.

    Raises:
        ToolArgumentError: If the arguments do not satisfy the schema.
    """
    from continuum.tools.exceptions import ToolArgumentError

    if not isinstance(schema, dict):
        return

    if not isinstance(arguments, dict):
        raise ToolArgumentError(
            f"Tool '{tool_name}' expects an object of arguments, got {type(arguments).__name__}"
        )

    properties = schema.get("properties")
    required = schema.get("required") or []

    missing = [name for name in required if name not in arguments]
    if missing:
        raise ToolArgumentError(
            f"Tool '{tool_name}' is missing required argument(s): {', '.join(sorted(missing))}"
        )

    # Unknown keys are a signal, not noise: a name the tool never declared is
    # either a hallucinated argument or an attempt to reach a parameter the
    # schema deliberately withheld. Only checked when the schema actually
    # enumerates its properties -- see the docstring.
    if isinstance(properties, dict) and schema.get("additionalProperties") is not True:
        unknown = [name for name in arguments if name not in properties]
        if unknown:
            raise ToolArgumentError(
                f"Tool '{tool_name}' received unknown argument(s): {', '.join(sorted(unknown))}. "
                f"Accepted: {', '.join(sorted(properties)) or '(none)'}"
            )

    if not isinstance(properties, dict):
        return

    for name, value in arguments.items():
        prop = properties.get(name)
        if not isinstance(prop, dict) or not prop:
            continue  # open schema -- no constraint was ever declared

        # A property absent from ``required`` may legitimately be omitted, and
        # JSON expresses omission as null: providers routinely send an explicit
        # null for an unfilled optional. Rejecting that would break working
        # tools for no gain -- the parameter's own default handles it.
        if value is None and name not in required:
            continue

        declared = prop.get("type")
        if declared is not None and not _type_matches(declared, value):
            expected = declared if isinstance(declared, str) else "/".join(declared)
            raise ToolArgumentError(
                f"Tool '{tool_name}' argument '{name}' must be {expected}, "
                f"got {type(value).__name__}"
            )

        choices = prop.get("enum")
        if isinstance(choices, list) and value not in choices:
            raise ToolArgumentError(
                f"Tool '{tool_name}' argument '{name}' must be one of "
                f"{', '.join(repr(c) for c in choices)}, got {value!r}"
            )
