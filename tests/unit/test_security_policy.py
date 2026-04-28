"""
Tests for the deny-overrides policy engine (security/policy.py).
"""
from __future__ import annotations

import pytest

from orchestrator.security.policy import AccessPolicy, PolicyDecision, PolicyStore


def _allow(name, subjects, resources):
    return AccessPolicy(name=name, subjects=subjects, resources=resources, effect="allow")


def _deny(name, subjects, resources):
    return AccessPolicy(name=name, subjects=subjects, resources=resources, effect="deny")


class TestPolicyDecision:
    def test_fields(self):
        d = PolicyDecision(allowed=True, policy_name="p1", reason="ok")
        assert d.allowed is True
        assert d.policy_name == "p1"
        assert d.reason == "ok"


class TestPolicyStoreOpenDefault:
    def test_no_policies_allows_everything(self):
        store = PolicyStore()
        result = store.check("any_agent", "tool:anything")
        assert result.allowed is True
        assert result.policy_name is None

    def test_reason_mentions_open_default(self):
        result = PolicyStore().check("agent", "tool:x")
        assert "open" in result.reason.lower() or "no matching" in result.reason.lower()


class TestPolicyStoreExplicitAllow:
    def test_matching_allow_is_allowed(self):
        store = PolicyStore()
        store.add_policy(_allow("p1", ["billing_agent"], ["tool:get_invoice"]))
        result = store.check("billing_agent", "tool:get_invoice")
        assert result.allowed is True
        assert result.policy_name == "p1"

    def test_non_matching_subject_falls_to_default(self):
        store = PolicyStore()
        store.add_policy(_allow("p1", ["billing_agent"], ["tool:get_invoice"]))
        result = store.check("other_agent", "tool:get_invoice")
        assert result.allowed is True  # open default
        assert result.policy_name is None

    def test_non_matching_resource_falls_to_default(self):
        store = PolicyStore()
        store.add_policy(_allow("p1", ["billing_agent"], ["tool:get_invoice"]))
        result = store.check("billing_agent", "tool:delete_user")
        assert result.allowed is True
        assert result.policy_name is None


class TestPolicyStoreDenyOverrides:
    def test_explicit_deny_blocks(self):
        store = PolicyStore()
        store.add_policy(_deny("block_shell", ["*"], ["tool:shell_*"]))
        result = store.check("admin_agent", "tool:shell_exec")
        assert result.allowed is False
        assert result.policy_name == "block_shell"

    def test_deny_beats_allow_same_resource(self):
        store = PolicyStore()
        store.add_policy(_allow("allow_all", ["*"], ["tool:*"]))
        store.add_policy(_deny("block_dangerous", ["*"], ["tool:shell_exec"]))
        result = store.check("agent", "tool:shell_exec")
        assert result.allowed is False

    def test_deny_does_not_affect_non_matching_resource(self):
        store = PolicyStore()
        store.add_policy(_deny("block_shell", ["*"], ["tool:shell_*"]))
        result = store.check("agent", "tool:get_weather")
        assert result.allowed is True


class TestPolicyStoreGlobMatching:
    def test_wildcard_subject(self):
        store = PolicyStore()
        store.add_policy(_deny("block_all_delete", ["*"], ["tool:delete_*"]))
        assert store.check("any_agent", "tool:delete_user").allowed is False
        assert store.check("any_agent", "tool:delete_order").allowed is False
        assert store.check("any_agent", "tool:get_user").allowed is True

    def test_specific_subject_pattern(self):
        store = PolicyStore()
        store.add_policy(_allow("billing_only", ["billing_*"], ["tool:invoice_*"]))
        assert store.check("billing_agent", "tool:invoice_read").allowed is True
        assert store.check("technical_agent", "tool:invoice_read").allowed is True  # open default

    def test_data_label_resource(self):
        store = PolicyStore()
        store.add_policy(_deny("no_pii", ["summarizer_agent"], ["data:pii"]))
        assert store.check("summarizer_agent", "data:pii").allowed is False
        assert store.check("summarizer_agent", "data:public").allowed is True


class TestPolicyStoreMutability:
    def test_add_replaces_existing_name(self):
        store = PolicyStore()
        store.add_policy(_allow("p1", ["agent_a"], ["tool:x"]))
        store.add_policy(_deny("p1", ["agent_a"], ["tool:x"]))  # replace
        assert len(store.list_policies()) == 1
        assert store.list_policies()[0].effect == "deny"

    def test_remove_existing(self):
        store = PolicyStore()
        store.add_policy(_deny("p1", ["*"], ["tool:dangerous"]))
        removed = store.remove_policy("p1")
        assert removed is True
        assert store.check("agent", "tool:dangerous").allowed is True

    def test_remove_nonexistent_returns_false(self):
        store = PolicyStore()
        assert store.remove_policy("ghost") is False

    def test_list_policies_returns_copy(self):
        store = PolicyStore()
        store.add_policy(_allow("p1", ["a"], ["b"]))
        lst = store.list_policies()
        lst.clear()
        assert len(store.list_policies()) == 1
