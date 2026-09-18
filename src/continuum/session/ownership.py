"""
Session ownership — the decision, kept separate from the plumbing.

One pure function decides whether a caller may touch a session, so the rule can
be read and tested without a store, a client or an event loop.

The rule, in the order it is applied::

    no principal, none required     -> allow      (capability access; see below)
    ownership unverifiable          -> a problem  (cannot tell, so do not guess)
    session has no stored owner     -> allow      (nothing to protect)
    no principal, one required      -> a problem
    principal == stored owner       -> allow
    principal != stored owner       -> a problem

The first rule comes first for a reason beyond brevity: when it applies, no
outcome could have been a refusal, so the caller need not read the stored owner
at all. That keeps the added cost at zero for deployments that have not adopted
principals, and stops a degraded store from failing closed over a check that
could never have refused anything.

"No principal bound" allowing access is a deliberate choice, and it is only
sound alongside the keyed session ids in :mod:`continuum.session.identity`.
Once an id cannot be constructed from a user id, holding one is reasonable
evidence of having been given it, and capability-style access is a defensible
default — which is what lets ownership checking arrive without breaking callers
that never adopted principals. Deployments that want that door shut anyway set
``require_principal``; they do not have to wait for the next major.

A stored owner of None is not the same as "cannot read the store". When
persistence is degraded every session looks unowned, so ``unverifiable`` marks
that case and ``enforce`` refuses instead of waving it through. Same reasoning
as the weak-Redis-password path, which fails closed rather than degrading.

Modes exist because this changes behaviour for existing deployments:

    open      today's behaviour — report a problem, allow it
    audit     report louder, with a metric, so breakage can be measured
    enforce   raise

Only ``enforce`` refuses. ``open`` and ``audit`` differ in how loudly they
report, which is the SessionClient's business, not this function's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

OwnershipMode = Literal["open", "audit", "enforce"]

# Problem codes. Stable strings — they are attached to log records and metrics,
# so a dashboard can distinguish "callers have not adopted principals yet" from
# "somebody is using another user's session id".
MISMATCH = "principal_mismatch"
NO_PRINCIPAL = "no_principal_bound"
UNVERIFIABLE = "ownership_unverifiable"


@dataclass(frozen=True)
class OwnershipCheck:
    """The outcome of one ownership evaluation.

    ``allowed`` and ``problem`` are independent on purpose: below ``enforce`` an
    access is allowed *and* problematic, which is exactly the state a team needs
    to see while deciding whether they can turn enforcement on.
    """

    allowed: bool
    problem: str | None = None

    @property
    def clean(self) -> bool:
        """True when the access raised no concern at all."""
        return self.allowed and self.problem is None


def evaluate_ownership(
    *,
    stored_user_id: str | None,
    principal: str | None,
    mode: OwnershipMode = "open",
    require_principal: bool = False,
    unverifiable: bool = False,
) -> OwnershipCheck:
    """Decide whether this caller may touch this session.

    Args:
        stored_user_id: owner recorded with the session; None means unowned.
        principal: verified caller identity; None means none was asserted.
        mode: 'open' and 'audit' report only, 'enforce' refuses.
        require_principal: treat an unasserted identity as a problem.
        unverifiable: the stored owner could not be read (degraded persistence),
            so ``stored_user_id`` carries no information.
    """
    problem = _find_problem(
        stored_user_id=stored_user_id,
        principal=principal,
        require_principal=require_principal,
        unverifiable=unverifiable,
    )
    if problem is None:
        return OwnershipCheck(allowed=True)
    return OwnershipCheck(allowed=mode != "enforce", problem=problem)


def _find_problem(
    *,
    stored_user_id: str | None,
    principal: str | None,
    require_principal: bool,
    unverifiable: bool,
) -> str | None:
    if principal is None and not require_principal:
        # This policy has nothing it could object to: with no asserted identity
        # to compare against and no requirement for one, every outcome is
        # "allow". Decided first so the caller can skip reading the stored owner
        # entirely — a deployment that has not adopted principals pays no extra
        # round-trip, and does not get failed closed on a degraded store over a
        # check that could never have refused.
        return None
    if unverifiable:
        # Cannot distinguish "no such session" from "cannot reach the store", so
        # do not read an absent owner as "unowned".
        return UNVERIFIABLE
    if stored_user_id is None:
        # Never bound to anyone: an anonymous or single-user deployment. There
        # is no owner to exclude.
        return None
    if principal is None:
        return NO_PRINCIPAL  # require_principal is True to have reached here
    if principal != stored_user_id:
        return MISMATCH
    return None


def describe(problem: str, session_id: str) -> str:
    """A refusal message for ``problem``.

    Never names the stored owner: the caller has already failed to prove they
    are that person, so telling them who it is would turn the refusal into an
    identity-disclosure oracle.
    """
    if problem == MISMATCH:
        return (
            f"session_id={session_id!r} belongs to a different principal. "
            "A session id names storage; it is not authorization on its own."
        )
    if problem == NO_PRINCIPAL:
        return (
            f"session_id={session_id!r} is owned, but no principal was bound for "
            "this call. Bind the caller's verified identity with "
            "continuum.session.bind_principal() at your auth boundary."
        )
    if problem == UNVERIFIABLE:
        return (
            f"ownership of session_id={session_id!r} could not be verified "
            "because session persistence is degraded. Refusing rather than "
            "assuming the session is unowned."
        )
    return f"session_id={session_id!r} failed the ownership check ({problem})."
