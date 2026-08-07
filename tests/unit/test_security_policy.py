"""
Tests for the deny-overrides policy engine (security/policy.py).
"""

from __future__ import annotations

from continuum.security.policy import AccessPolicy, PolicyDecision, PolicyStore


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


class TestPolicyStoreMultiSubject:
    """`check()` accepts multiple subjects — this is how RunContext.data_labels
    gate access: a data label tainting the run acts as an extra subject."""

    def test_single_string_subject_still_works(self):
        store = PolicyStore()
        store.add_policy(_deny("no_pii_email", ["pii"], ["tool:send_email"]))
        # Backward compatible: plain string subject, no label → allowed.
        assert store.check("agent", "tool:send_email").allowed is True

    def test_label_in_subjects_triggers_deny(self):
        store = PolicyStore()
        store.add_policy(_deny("no_pii_email", ["pii"], ["tool:send_email"]))
        # Agent name alone → allowed; agent + "pii" label → denied.
        assert store.check(["agent"], "tool:send_email").allowed is True
        decision = store.check(["agent", "pii"], "tool:send_email")
        assert decision.allowed is False
        assert decision.policy_name == "no_pii_email"

    def test_label_only_denies_matching_resource(self):
        store = PolicyStore()
        store.add_policy(_deny("no_pii_email", ["pii"], ["tool:send_email"]))
        # A different tool is unaffected even when the pii label is present.
        assert store.check(["agent", "pii"], "tool:lookup_account").allowed is True

    def test_any_subject_match_is_enough(self):
        store = PolicyStore()
        store.add_policy(_deny("block", ["phi"], ["tool:*"]))
        # Only one of several subjects needs to match the policy.
        assert store.check(["agent", "pii", "phi"], "tool:anything").allowed is False
        assert store.check(["agent", "pii"], "tool:anything").allowed is True

    def test_empty_extra_subjects_behaves_like_agent_only(self):
        store = PolicyStore()
        store.add_policy(_deny("no_pii_email", ["pii"], ["tool:send_email"]))
        assert store.check(["agent"], "tool:send_email").allowed is True


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


class TestPolicyStoreDefaultEffect:
    """Step 3 — opt-in default-deny posture when a store IS configured."""

    def test_default_is_allow_backward_compatible(self):
        # Unset default_effect must preserve the historical open-default behavior.
        store = PolicyStore()
        assert store.default_effect == "allow"
        assert store.check("agent", "tool:anything").allowed is True

    def test_default_deny_blocks_unmatched(self):
        store = PolicyStore(default_effect="deny")
        decision = store.check("agent", "tool:delete_account")
        assert decision.allowed is False
        assert decision.policy_name is None
        assert "default" in decision.reason.lower()

    def test_default_deny_allows_explicitly_allowed(self):
        store = PolicyStore(default_effect="deny")
        store.add_policy(_allow("p1", ["billing_agent"], ["tool:get_invoice"]))
        assert store.check("billing_agent", "tool:get_invoice").allowed is True
        # Anything not explicitly allowed is blocked.
        assert store.check("billing_agent", "tool:delete_account").allowed is False

    def test_default_deny_still_honours_explicit_deny(self):
        # deny-overrides semantics remain intact under default-deny.
        store = PolicyStore(default_effect="deny")
        store.add_policy(_allow("all", ["*"], ["tool:*"]))
        store.add_policy(_deny("no_shell", ["*"], ["tool:shell_exec"]))
        assert store.check("agent", "tool:shell_exec").allowed is False
        assert store.check("agent", "tool:get_weather").allowed is True


class TestPolicyStoreDefaultDenyFactory:
    """Step 2 — ergonomic secure-defaults constructor for the quick-start path."""

    def test_default_deny_factory_denies_by_default(self):
        store = PolicyStore.default_deny()
        assert store.default_effect == "deny"
        assert store.check("agent", "tool:anything").allowed is False

    def test_default_deny_factory_seeds_allow_policies(self):
        store = PolicyStore.default_deny([_allow("read_only", ["*"], ["tool:get_*"])])
        assert store.check("agent", "tool:get_invoice").allowed is True
        assert store.check("agent", "tool:delete_invoice").allowed is False
