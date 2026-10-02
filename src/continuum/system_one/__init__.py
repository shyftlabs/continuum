"""System One classifiers: fast, typed judgements seams can opt into.

A System One classifier (Kahneman's sense: fast, cheap judgement, not the slow
reasoning of an LLM) answers typed questions about a state with a probability
distribution per question. Jev is one backend; local open-source classifiers are
another; third parties register more through the ``continuum.system_one`` entry
point group.

Nothing here is on by default. Each seam -- ``RouterAgent``, ``LoopAgent``, the
tool-approval handler -- opts in on its own, and ``SYSTEM_ONE_DISABLED=1`` sends
every seam back to its previous behaviour.

    from continuum.system_one import BinaryQuestion, classify

    resp = await classify(
        {"tool": "transfer_funds", "arguments": {"amount": 5_000_000}},
        {"risky": BinaryQuestion(instructions="Would executing this call be risky?")},
        spec="jev:jev-latest",
    )
    resp.answers["risky"].probability
"""

from continuum.protocols import ISystemOneClassifier
from continuum.system_one.exceptions import (
    SystemOneAccessDeniedError,
    SystemOneBackendError,
    SystemOneCapabilityError,
    SystemOneDisabledError,
    SystemOneError,
    SystemOneNotConfiguredError,
    SystemOneResponseError,
    SystemOneTimeoutError,
)
from continuum.system_one.layer import classify, egress_resource, system_one_disabled
from continuum.system_one.registry import (
    ENTRY_POINT_GROUP,
    create_classifier,
    register_backend,
    require_backend,
    resolve_classifier,
)
from continuum.system_one.types import (
    Answer,
    BinaryAnswer,
    BinaryQuestion,
    ChoiceAnswer,
    ChoiceQuestion,
    Provenance,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    State,
    SystemOneCapabilities,
    SystemOneRawResult,
    SystemOneResponse,
)

__all__ = [
    "ENTRY_POINT_GROUP",
    "Answer",
    "BinaryAnswer",
    "BinaryQuestion",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "ISystemOneClassifier",
    "Provenance",
    "Question",
    "ScoreAnswer",
    "ScoreQuestion",
    "State",
    "SystemOneAccessDeniedError",
    "SystemOneBackendError",
    "SystemOneCapabilities",
    "SystemOneCapabilityError",
    "SystemOneDisabledError",
    "SystemOneError",
    "SystemOneNotConfiguredError",
    "SystemOneRawResult",
    "SystemOneResponse",
    "SystemOneResponseError",
    "SystemOneTimeoutError",
    "classify",
    "create_classifier",
    "egress_resource",
    "register_backend",
    "require_backend",
    "resolve_classifier",
    "system_one_disabled",
]
