"""
Tests for new RunContext fields: conversation_id and is_handoff.
"""
from __future__ import annotations

import pytest

from orchestrator.agent.types import RunContext, PrepareRunResult
from orchestrator.agent.utils.context_utils import create_run_context


class TestRunContextConversationId:
    def test_defaults_to_none(self):
        ctx = RunContext(run_id="r1")
        assert ctx.conversation_id is None

    def test_set_on_construction(self):
        ctx = RunContext(run_id="r1", conversation_id="conv-123")
        assert ctx.conversation_id == "conv-123"

    def test_included_in_to_dict(self):
        ctx = RunContext(run_id="r1", conversation_id="conv-abc")
        d = ctx.to_dict()
        assert d["conversation_id"] == "conv-abc"

    def test_none_included_in_to_dict(self):
        ctx = RunContext(run_id="r1")
        d = ctx.to_dict()
        assert "conversation_id" in d
        assert d["conversation_id"] is None


class TestRunContextIsHandoff:
    def test_defaults_to_false(self):
        ctx = RunContext(run_id="r1")
        assert ctx.is_handoff is False

    def test_set_true_on_construction(self):
        ctx = RunContext(run_id="r1", is_handoff=True)
        assert ctx.is_handoff is True

    def test_is_handoff_is_transient_flag(self):
        # is_handoff is a runtime execution flag, not serialized to to_dict
        ctx = RunContext(run_id="r1", is_handoff=True)
        assert ctx.is_handoff is True
        d = ctx.to_dict()
        # conversation_id IS in to_dict, is_handoff is not (it's transient)
        assert "conversation_id" in d
        assert "is_handoff" not in d


class TestCreateRunContext:
    def test_accepts_conversation_id(self):
        ctx = create_run_context(conversation_id="conv-456")
        assert ctx.conversation_id == "conv-456"

    def test_conversation_id_defaults_to_none(self):
        ctx = create_run_context()
        assert ctx.conversation_id is None

    def test_is_handoff_defaults_to_false(self):
        ctx = create_run_context()
        assert ctx.is_handoff is False

    def test_all_fields_together(self):
        ctx = create_run_context(
            session_id="sess-1",
            conversation_id="conv-1",
            user_id="user-1",
        )
        assert ctx.session_id == "sess-1"
        assert ctx.conversation_id == "conv-1"
        assert ctx.user_id == "user-1"
        assert ctx.is_handoff is False


class TestPrepareRunResultUserMessageIndex:
    def test_default_is_zero(self):
        result = PrepareRunResult(success=True)
        assert result.user_message_index == 0

    def test_set_explicitly(self):
        result = PrepareRunResult(success=True, user_message_index=5)
        assert result.user_message_index == 5
