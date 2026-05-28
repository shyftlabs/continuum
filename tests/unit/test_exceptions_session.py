"""
Tests for exceptions added/fixed this session:
- InputBlockedError: was never defined (live crash bug)
- ValidationError: __init__ was accidentally deleted (silent bug)
"""

from __future__ import annotations

import pytest

from orchestrator.exceptions import (
    InputBlockedError,
    OrchestratorError,
    ValidationError,
)


class TestInputBlockedError:
    def test_is_importable(self):
        assert InputBlockedError is not None

    def test_is_orchestrator_error(self):
        assert issubclass(InputBlockedError, OrchestratorError)

    def test_raises_correctly(self):
        with pytest.raises(InputBlockedError):
            raise InputBlockedError("blocked")

    def test_default_message(self):
        err = InputBlockedError()
        assert "blocked" in err.message.lower() or "input" in err.message.lower()

    def test_reason_in_context(self):
        err = InputBlockedError(reason="prompt injection detected", scanner="injection_scanner")
        assert err.context["reason"] == "prompt injection detected"
        assert err.context["scanner"] == "injection_scanner"

    def test_error_code(self):
        err = InputBlockedError()
        assert err.error_code == "INPUT_BLOCKED"

    def test_custom_message(self):
        err = InputBlockedError("custom block message")
        assert err.message == "custom block message"


class TestValidationError:
    def test_init_accepts_message(self):
        err = ValidationError("bad input")
        assert err.message == "bad input"

    def test_field_stored_in_context(self):
        err = ValidationError("missing field", field="email")
        assert err.context["field"] == "email"

    def test_value_stored_in_context(self):
        err = ValidationError("wrong value", value="bad_val")
        assert err.context["value"] == "bad_val"

    def test_expected_stored_in_context(self):
        err = ValidationError("type mismatch", expected="int")
        assert err.context["expected"] == "int"

    def test_all_fields_together(self):
        err = ValidationError("fail", field="age", value=-1, expected="positive int")
        assert err.context["field"] == "age"
        assert err.context["value"] == "-1"
        assert err.context["expected"] == "positive int"

    def test_no_args(self):
        err = ValidationError()
        assert err.message == ValidationError.default_message

    def test_error_code(self):
        err = ValidationError()
        assert err.error_code == "VALIDATION_ERROR"

    def test_is_orchestrator_error(self):
        assert issubclass(ValidationError, OrchestratorError)
