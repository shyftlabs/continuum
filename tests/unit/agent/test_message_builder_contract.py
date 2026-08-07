"""
IMessageBuilder's declared return type must match what the runner unpacks.

`IMessageBuilder` is exported from `continuum.agent`, so it is the contract a
third party writes a custom message builder against. It declared
`-> list[dict[str, Any]]` while `MessageBuilder` returns
`(messages, user_message_index)` and `AgentRunner` unpacks two values. Anyone
implementing the published interface faithfully would have returned a bare list
and broken the runner on the first call — the internal implementation was the
only thing keeping this working.

Signature-level tests: no LLM, no runner execution.
"""

from __future__ import annotations

import inspect

from continuum.agent.execution.message_builder import MessageBuilder
from continuum.agent.interfaces.handler_interface import IMessageBuilder


def _return_annotation(cls: type, method: str) -> str:
    """Raw annotation text.

    Both modules use `from __future__ import annotations` with TYPE_CHECKING-only
    imports, so `get_type_hints` cannot resolve them at runtime — compare the
    declared source text instead.
    """
    return str(inspect.signature(getattr(cls, method)).return_annotation)


class TestPrepareMessagesContract:
    def test_interface_and_implementation_agree(self) -> None:
        assert _return_annotation(IMessageBuilder, "prepare_messages") == _return_annotation(
            MessageBuilder, "prepare_messages"
        )

    def test_interface_declares_a_two_tuple(self) -> None:
        """The runner does `messages, user_message_index = await ...`, so the
        contract must promise exactly two values, the second an int index."""
        annotation = _return_annotation(IMessageBuilder, "prepare_messages")
        assert annotation.startswith("tuple["), f"expected a tuple, got {annotation}"
        assert annotation.endswith(", int]"), f"expected an int index, got {annotation}"

    def test_parameters_still_match(self) -> None:
        """Guard the rest of the signature too, not just the return type."""
        iface = inspect.signature(IMessageBuilder.prepare_messages).parameters
        impl = inspect.signature(MessageBuilder.prepare_messages).parameters
        assert list(iface) == list(impl)
