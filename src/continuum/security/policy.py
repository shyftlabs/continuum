"""Deny-overrides access control policy engine.

Inspired by Orla's serving/access/evaluator.go:
  - Subjects are agent names or data labels (glob patterns).
  - Resources are tool names, memory scopes, or data labels (glob patterns).
  - Deny always overrides allow (explicit deny wins).
  - If no policy matches, the result is PolicyStore.default_effect
    ("allow" by default — historical open default; set "deny", or use
    PolicyStore.default_deny(), to fail closed).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from fnmatch import fnmatch


@dataclass
class AccessPolicy:
    """A single access control rule.

    Attributes:
        name: Unique identifier for the policy.
        subjects: Glob patterns matched against the caller identity
            (e.g. agent name, tag value like "billing_agent").
        resources: Glob patterns matched against the resource being
            accessed (e.g. "tool:delete_*", "memory:user_*", "data:pii").
        effect: "allow" or "deny". Deny overrides allow when both match.
    """

    name: str
    subjects: list[str]
    resources: list[str]
    effect: str  # "allow" | "deny"
    denial_message: str = ""  # shown to LLM when this policy denies a request


@dataclass
class PolicyDecision:
    """Result of an access check."""

    allowed: bool
    policy_name: str | None = None
    reason: str = ""
    denial_message: str = ""


@dataclass
class PolicyStore:
    """In-memory policy store with deny-overrides evaluation.

    Evaluation order (mirrors Orla's evaluator.go CheckAccess):
      1. Any matching explicit deny → DENY.
      2. Any matching explicit allow → ALLOW.
      3. No match → ``default_effect`` (``"allow"`` by default, historical
         open-default behavior; set ``"deny"`` for a secure default-deny
         posture where anything not explicitly allowed is blocked).

    Example (secure default-deny):
        store = PolicyStore.default_deny([
            AccessPolicy(name="read", subjects=["*"],
                         resources=["tool:get_*"], effect="allow"),
        ])

    Tool resources are the LLM-facing tool name. **MCP** tools are namespaced by
    default, so theirs is ``tool:<server>__<name>`` -- scope a whole server with
    ``tool:weather__*``, or span both shapes with ``tool:*get_*``. Local function
    tools keep bare names. This matters most for ``deny`` rules: with the default
    ``default_effect="allow"``, a rule that matches nothing stops blocking rather
    than failing closed. See docs/tools.md section 6.5.
    """

    _policies: list[AccessPolicy] = field(default_factory=list)
    # Result when no policy matches. "allow" preserves the original open
    # default; "deny" fails closed so a forgotten rule blocks instead of opens.
    default_effect: str = "allow"

    @classmethod
    def default_deny(cls, policies: list[AccessPolicy] | None = None) -> PolicyStore:
        """Build a fail-closed store: anything not explicitly allowed is denied.

        This is the recommended secure-defaults constructor for the quick-start
        path — seed it with the explicit allow policies your agent needs.
        """
        store = cls(default_effect="deny")
        for policy in policies or []:
            store.add_policy(policy)
        return store

    def add_policy(self, policy: AccessPolicy) -> None:
        """Add a policy to the store. Replaces any existing policy with the same name."""
        self._policies = [p for p in self._policies if p.name != policy.name]
        self._policies.append(policy)

    def remove_policy(self, name: str) -> bool:
        """Remove a policy by name. Returns True if it existed."""
        before = len(self._policies)
        self._policies = [p for p in self._policies if p.name != name]
        return len(self._policies) < before

    def list_policies(self) -> list[AccessPolicy]:
        return list(self._policies)

    def check(self, subject: str | Iterable[str], resource: str) -> PolicyDecision:
        """Evaluate policies for (subject, resource).

        Args:
            subject: Caller identity — typically the agent name. May also be an
                iterable of identities (e.g. the agent name plus the run's active
                data-sensitivity labels like ``"pii"``); a policy matches if ANY
                of them matches its ``subjects`` globs. This is how
                ``RunContext.data_labels`` gates access: a label tainting the run
                acts as an additional subject.
            resource: Resource being accessed, e.g. "tool:shell_exec",
                "memory:user_notes", "data:pii".

        Returns:
            PolicyDecision with allowed=True/False and the matching policy name.
        """
        subjects = [subject] if isinstance(subject, str) else list(subject)

        deny_match: AccessPolicy | None = None
        allow_match: AccessPolicy | None = None

        for policy in self._policies:
            if not any(_matches_any(s, policy.subjects) for s in subjects):
                continue
            if not _matches_any(resource, policy.resources):
                continue
            if policy.effect == "deny":
                deny_match = policy
                break  # Deny found — no need to keep searching
            elif policy.effect == "allow" and allow_match is None:
                allow_match = policy

        if deny_match is not None:
            return PolicyDecision(
                allowed=False,
                policy_name=deny_match.name,
                reason=f"explicit deny by policy '{deny_match.name}'",
                denial_message=deny_match.denial_message,
            )
        if allow_match is not None:
            return PolicyDecision(
                allowed=True,
                policy_name=allow_match.name,
                reason=f"explicit allow by policy '{allow_match.name}'",
            )
        if self.default_effect == "deny":
            return PolicyDecision(
                allowed=False,
                reason="no matching policy (closed default — default-deny)",
                denial_message="No policy explicitly allows this access (default-deny).",
            )
        return PolicyDecision(allowed=True, reason="no matching policy (open default)")


def _matches_any(value: str, patterns: list[str]) -> bool:
    return any(fnmatch(value, pat) for pat in patterns)
