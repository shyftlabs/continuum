"""
Span payloads must be inert data, never live objects.

Two defects composed into an OOMKill in a downstream service (v1.2.0): every
decorated method captured `self`, and the size guard measured a stand-in for
the object rather than the object, then returned the object it had never
actually weighed. A ToolExecutor whose repr measured 49 bytes expanded to
93,772 bytes once the telemetry backend walked its __dict__.

Both halves are tested independently: either one alone is enough to put a live
object on the wire, so neither may regress on the strength of the other.
"""

from __future__ import annotations

import json

from continuum.observability.decorators import _get_function_input, observe
from continuum.observability.trace_context import MAX_TRACE_DATA_SIZE, truncate_data


class Live:
    """Stand-in for a heavy runtime object (ToolExecutor, LLMClient, ...).

    Self-referential on purpose: a naive deep copy or recursive walk of this
    must not be what protects us.
    """

    def __init__(self, payload: str = "x"):
        self.peer = self
        self.payload = payload


class TestGetFunctionInputDropsSelf:
    """Defect 1 — the capture helper bound `self` along with the arguments."""

    def test_self_is_not_captured_from_a_bound_method(self):
        class Service:
            def handle(self, job_id: str, retries: int = 0) -> None: ...

        captured = _get_function_input(Service.handle, (Service(), "job-1"), {})

        assert "self" not in captured
        assert captured == {"job_id": "job-1", "retries": 0}

    def test_cls_is_not_captured_from_a_classmethod(self):
        class Service:
            @classmethod
            def build(cls, name: str) -> None: ...

        captured = _get_function_input(Service.build.__func__, (Service, "svc"), {})

        assert "cls" not in captured
        assert captured == {"name": "svc"}

    def test_plain_function_arguments_are_untouched(self):
        # The fix must key on the receiver position, not on the name — a plain
        # function is entitled to a parameter called `self`.
        def transform(self, data: dict) -> None: ...

        captured = _get_function_input(transform, ("not-a-receiver", {"k": "v"}), {})

        assert captured == {"self": "not-a-receiver", "data": {"k": "v"}}

    def test_decorated_method_puts_no_self_on_the_span(self):
        seen: dict = {}

        class Service:
            def __init__(self):
                self.registry = {"tool": "definition"}

            @observe(name="svc_call", capture_input=True)
            def call(self, job_id: str) -> str:
                return "done"

        # SpanScope no-ops without a trace context, so assert on what the
        # decorator built rather than on what was exported.
        original = _get_function_input

        def spy(func, args, kwargs):
            seen.update(result := original(func, args, kwargs))
            return result

        import continuum.observability.decorators as decorators

        decorators._get_function_input = spy
        try:
            Service().call("job-1")
        finally:
            decorators._get_function_input = original

        assert "self" not in seen
        assert seen == {"job_id": "job-1"}


class TestTruncateDataReturnsWhatItMeasured:
    """Defect 2 — the guard returned the original, not the flattened form."""

    def test_live_object_does_not_survive(self):
        out = truncate_data({"self": Live()})

        assert not isinstance(out["self"], Live)
        assert isinstance(out["self"], str)
        assert "Live object at" in out["self"]

    def test_return_value_is_json_serializable_without_a_default(self):
        # The real defect: the returned value had never been proven encodable.
        # json.dumps with no `default` is exactly the test the backend applies.
        out = truncate_data({"executor": Live(), "job_id": "job-1"})

        json.dumps(out)  # must not raise

    def test_small_plain_data_is_returned_faithfully(self):
        data = {"messages": [{"role": "user", "content": "hi"}], "n": 3, "ok": True}

        assert truncate_data(data) == data

    def test_oversized_data_still_truncates(self):
        data = {"blob": "x" * (MAX_TRACE_DATA_SIZE * 2)}

        out = truncate_data(data)

        assert out["_truncated"] is True
        assert out["_original_size"] > MAX_TRACE_DATA_SIZE

    def test_unencodable_data_still_falls_back_to_a_string(self):
        # Keys json cannot encode at all — the existing except branch owns this
        # and must keep owning it.
        out = truncate_data({("tuple", "key"): Live()})

        assert isinstance(out, str)

    def test_nested_live_object_does_not_survive(self):
        # Dropping `self` alone would not catch this: a caller can pass a live
        # object as an ordinary argument.
        out = truncate_data({"ctx": {"deep": [Live()]}})

        json.dumps(out)  # must not raise

    def test_none_is_preserved(self):
        assert truncate_data(None) is None
