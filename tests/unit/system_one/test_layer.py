"""The shared System One layer: what a seam gets back, whatever backend answered.

Backends differ in which question kinds they support, whether they accept JSON
state, and how they express confidence. The layer is the one place that evens
those out, so a seam written against it behaves the same on every backend.
"""

from __future__ import annotations

import math

import pytest


def _binary(text: str = "Is it urgent?"):
    from continuum.system_one import BinaryQuestion

    return BinaryQuestion(instructions=text)


def _choice(labels=("billing", "technical", "none")):
    from continuum.system_one import ChoiceQuestion

    return ChoiceQuestion(instructions="Which team?", labels=dict.fromkeys(labels))


def _score(levels=("calm", "concerned", "angry")):
    from continuum.system_one import ScoreQuestion

    return ScoreQuestion(instructions="How frustrated?", levels=list(levels))


# ---------------------------------------------------------------------------
# Question types
# ---------------------------------------------------------------------------


class TestQuestionTypesRejectWhatCannotBeAnswered:
    def test_a_choice_needs_at_least_two_labels(self):
        """One label is not a choice: the answer is known before asking."""
        from continuum.system_one import ChoiceQuestion

        with pytest.raises(ValueError):
            ChoiceQuestion(instructions="Pick", labels={"only": None})

    def test_a_score_needs_at_least_two_levels(self):
        from continuum.system_one import ScoreQuestion

        with pytest.raises(ValueError):
            ScoreQuestion(instructions="Rate", levels=["one"])

    def test_instructions_must_not_be_empty(self):
        from continuum.system_one import BinaryQuestion

        with pytest.raises(ValueError):
            BinaryQuestion(instructions="   ")

    def test_labels_are_copied_so_the_caller_cannot_edit_a_sent_question(self):
        from continuum.system_one import ChoiceQuestion

        labels = {"a": None, "b": None}
        q = ChoiceQuestion(instructions="Pick", labels=labels)
        labels["c"] = None
        assert list(q.labels) == ["a", "b"]


# ---------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------


class TestAnswers:
    async def test_a_binary_answer_carries_the_probability_of_true(self, fake_classifier_cls):
        from continuum.system_one import classify

        backend = fake_classifier_cls(answer=lambda qid, q: {"true": 0.9, "false": 0.1})
        resp = await classify("state", {"urgent": _binary()}, classifier=backend)

        answer = resp.answers["urgent"]
        assert answer.kind == "binary"
        assert answer.probability == pytest.approx(0.9)
        assert answer.probabilities == {"true": pytest.approx(0.9), "false": pytest.approx(0.1)}

    async def test_a_choice_answer_selects_the_most_probable_label(self, fake_classifier_cls):
        from continuum.system_one import classify

        backend = fake_classifier_cls(
            answer=lambda qid, q: {"billing": 0.2, "technical": 0.7, "none": 0.1}
        )
        resp = await classify("state", {"team": _choice()}, classifier=backend)

        assert resp.answers["team"].label == "technical"
        assert resp.answers["team"].probabilities["technical"] == pytest.approx(0.7)

    async def test_a_score_answer_is_the_expected_level(self, fake_classifier_cls):
        """The expected value, not the most likely level: 0.1*0 + 0.55*1 + 0.35*2."""
        from continuum.system_one import classify

        backend = fake_classifier_cls(answer=lambda qid, q: {0: 0.1, 1: 0.55, 2: 0.35})
        resp = await classify("state", {"frustration": _score()}, classifier=backend)

        assert resp.answers["frustration"].expected == pytest.approx(1.25)
        assert resp.answers["frustration"].probabilities == {
            0: pytest.approx(0.1),
            1: pytest.approx(0.55),
            2: pytest.approx(0.35),
        }

    async def test_a_distribution_that_does_not_sum_to_one_is_normalised(self, fake_classifier_cls):
        from continuum.system_one import classify

        backend = fake_classifier_cls(
            answer=lambda qid, q: {"billing": 2.0, "technical": 6.0, "none": 2.0}
        )
        resp = await classify("state", {"team": _choice()}, classifier=backend)

        probs = resp.answers["team"].probabilities
        assert sum(probs.values()) == pytest.approx(1.0)
        assert probs["technical"] == pytest.approx(0.6)


class TestConfidenceMeansTheSameOnEveryBackend:
    async def test_an_even_split_has_zero_confidence(self, fake_classifier_cls):
        from continuum.system_one import classify

        backend = fake_classifier_cls(answer=lambda qid, q: {"true": 0.5, "false": 0.5})
        resp = await classify("s", {"q": _binary()}, classifier=backend)
        assert resp.answers["q"].confidence == pytest.approx(0.0)

    async def test_a_certain_answer_has_full_confidence(self, fake_classifier_cls):
        from continuum.system_one import classify

        backend = fake_classifier_cls(answer=lambda qid, q: {"true": 1.0, "false": 0.0})
        resp = await classify("s", {"q": _binary()}, classifier=backend)
        assert resp.answers["q"].confidence == pytest.approx(1.0)

    async def test_confidence_is_one_minus_normalised_entropy(self, fake_classifier_cls):
        from continuum.system_one import classify

        dist = {"billing": 0.7, "technical": 0.2, "none": 0.1}
        backend = fake_classifier_cls(answer=lambda qid, q: dist)
        resp = await classify("s", {"q": _choice()}, classifier=backend)

        entropy = -sum(p * math.log(p) for p in dist.values())
        assert resp.answers["q"].confidence == pytest.approx(1 - entropy / math.log(3))

    async def test_the_backends_own_confidence_is_kept_separately(self, fake_classifier_cls):
        """Opaque and vendor-specific, so it never replaces Continuum's own."""
        from continuum.system_one import classify

        backend = fake_classifier_cls(
            answer=lambda qid, q: {"true": 0.5, "false": 0.5}, raw_confidence=0.93
        )
        resp = await classify("s", {"q": _binary()}, classifier=backend)
        assert resp.answers["q"].raw_confidence == pytest.approx(0.93)
        assert resp.answers["q"].confidence == pytest.approx(0.0)


class TestABackendAnswerThatCannotBeTrustedIsRefused:
    async def test_a_missing_answer_is_an_error(self, fake_classifier_cls):
        from continuum.system_one import SystemOneResponseError, classify

        class Forgetful(fake_classifier_cls):
            async def classify(self, state, questions):
                raw = await super().classify(state, questions)
                raw.distributions.pop("b")
                return raw

        with pytest.raises(SystemOneResponseError):
            await classify("s", {"a": _binary(), "b": _binary()}, classifier=Forgetful())

    async def test_a_probability_outside_zero_to_one_is_an_error(self, fake_classifier_cls):
        from continuum.system_one import SystemOneResponseError, classify

        backend = fake_classifier_cls(answer=lambda qid, q: {"true": 1.4, "false": -0.4})
        with pytest.raises(SystemOneResponseError):
            await classify("s", {"q": _binary()}, classifier=backend)

    async def test_a_nan_probability_is_an_error(self, fake_classifier_cls):
        from continuum.system_one import SystemOneResponseError, classify

        backend = fake_classifier_cls(answer=lambda qid, q: {"true": math.nan, "false": 0.5})
        with pytest.raises(SystemOneResponseError):
            await classify("s", {"q": _binary()}, classifier=backend)

    async def test_an_all_zero_distribution_is_an_error(self, fake_classifier_cls):
        from continuum.system_one import SystemOneResponseError, classify

        backend = fake_classifier_cls(
            answer=lambda qid, q: {"billing": 0.0, "technical": 0.0, "none": 0.0}
        )
        with pytest.raises(SystemOneResponseError):
            await classify("s", {"q": _choice()}, classifier=backend)

    async def test_a_label_that_was_not_offered_is_an_error(self, fake_classifier_cls):
        from continuum.system_one import SystemOneResponseError, classify

        backend = fake_classifier_cls(answer=lambda qid, q: {"billing": 0.5, "sales": 0.5})
        with pytest.raises(SystemOneResponseError):
            await classify("s", {"q": _choice()}, classifier=backend)

    async def test_no_questions_is_refused_before_any_call(self, fake_classifier_cls):
        from continuum.system_one import classify

        backend = fake_classifier_cls()
        with pytest.raises(ValueError):
            await classify("s", {}, classifier=backend)
        assert backend.calls == []


# ---------------------------------------------------------------------------
# Filling in question kinds a backend lacks
# ---------------------------------------------------------------------------


class TestMissingKindsAreFilledInOnce:
    async def test_a_score_is_asked_as_a_choice_over_its_levels(self, fake_classifier_cls):
        from continuum.system_one import classify

        backend = fake_classifier_cls(
            kinds=frozenset({"binary", "choice"}),
            answer=lambda qid, q: {"0": 0.1, "1": 0.55, "2": 0.35},
        )
        resp = await classify("s", {"f": _score()}, classifier=backend)

        sent = backend.calls[0][1]
        assert [q.kind for q in sent.values()] == ["choice"]
        answer = resp.answers["f"]
        assert answer.kind == "score"
        assert answer.expected == pytest.approx(1.25)
        assert answer.filled_in is True

    async def test_a_choice_is_asked_as_one_binary_per_label(self, fake_classifier_cls):
        from continuum.system_one import classify

        yes = {"billing": 0.2, "technical": 0.6, "none": 0.2}

        def answer(qid, q):
            label = qid.rsplit("::", 1)[1]
            return {"true": yes[label], "false": 1 - yes[label]}

        backend = fake_classifier_cls(kinds=frozenset({"binary"}), answer=answer)
        resp = await classify("s", {"team": _choice()}, classifier=backend)

        sent = backend.calls[0][1]
        assert len(sent) == 3 and all(q.kind == "binary" for q in sent.values())
        answer = resp.answers["team"]
        assert answer.kind == "choice"
        assert answer.label == "technical"
        assert sum(answer.probabilities.values()) == pytest.approx(1.0)
        assert answer.filled_in is True

    async def test_a_score_on_a_binary_only_backend_goes_through_both_steps(
        self, fake_classifier_cls
    ):
        from continuum.system_one import classify

        def answer(qid, q):
            level = int(qid.rsplit("::", 1)[1])
            p = {0: 0.0, 1: 0.0, 2: 1.0}[level]
            return {"true": p, "false": 1 - p}

        backend = fake_classifier_cls(kinds=frozenset({"binary"}), answer=answer)
        resp = await classify("s", {"f": _score()}, classifier=backend)

        assert resp.answers["f"].expected == pytest.approx(2.0)
        assert resp.answers["f"].filled_in is True

    async def test_native_answers_are_not_flagged(self, fake_classifier_cls):
        from continuum.system_one import classify

        resp = await classify("s", {"q": _binary()}, classifier=fake_classifier_cls())
        assert resp.answers["q"].filled_in is False

    async def test_a_kind_that_cannot_be_derived_is_refused(self, fake_classifier_cls):
        """A choice-only backend cannot answer a binary question: nothing it
        supports can be decomposed into one without inventing semantics."""
        from continuum.system_one import SystemOneCapabilityError, classify

        backend = fake_classifier_cls(kinds=frozenset({"choice"}))
        with pytest.raises(SystemOneCapabilityError):
            await classify("s", {"q": _binary()}, classifier=backend)
        assert backend.calls == []


class TestStateAndBatching:
    async def test_a_text_only_backend_receives_json_text(self, fake_classifier_cls):
        from continuum.system_one import classify

        backend = fake_classifier_cls(structured_state=False)
        await classify({"tool": "transfer", "amount": 5}, {"q": _binary()}, classifier=backend)

        state = backend.calls[0][0]
        assert isinstance(state, str)
        assert '"amount": 5' in state

    async def test_a_structured_backend_receives_the_state_unchanged(self, fake_classifier_cls):
        from continuum.system_one import classify

        backend = fake_classifier_cls(structured_state=True)
        payload = {"tool": "transfer", "amount": 5}
        await classify(payload, {"q": _binary()}, classifier=backend)
        assert backend.calls[0][0] == payload

    async def test_questions_beyond_max_questions_go_in_further_calls(self, fake_classifier_cls):
        from continuum.system_one import classify

        backend = fake_classifier_cls(max_questions=2)
        resp = await classify(
            "s", {"a": _binary(), "b": _binary(), "c": _binary()}, classifier=backend
        )
        assert len(backend.calls) == 2
        assert set(resp.answers) == {"a", "b", "c"}


class TestProvenance:
    async def test_the_response_says_which_backend_and_model_answered(self, fake_classifier_cls):
        from continuum.system_one import classify

        backend = fake_classifier_cls(model="fake-7", usage={"input_tokens": 12})
        resp = await classify("s", {"q": _binary()}, classifier=backend)

        assert resp.provenance.backend == "fake"
        assert resp.provenance.model == "fake-7"
        assert resp.provenance.latency_ms >= 0
        assert resp.provenance.usage == {"input_tokens": 12}


# ---------------------------------------------------------------------------
# The kill switch and the egress check
# ---------------------------------------------------------------------------


class TestTheKillSwitch:
    async def test_disabled_refuses_without_calling_the_backend(
        self, fake_classifier_cls, monkeypatch
    ):
        """A direct call must not bypass the switch ops flipped during an incident."""
        from continuum.config import settings
        from continuum.system_one import SystemOneDisabledError, classify

        monkeypatch.setattr(settings, "system_one_disabled", True)
        backend = fake_classifier_cls()
        with pytest.raises(SystemOneDisabledError):
            await classify("s", {"q": _binary()}, classifier=backend)
        assert backend.calls == []

    def test_the_switch_is_read_live(self, monkeypatch):
        from continuum.config import settings
        from continuum.system_one import system_one_disabled

        assert system_one_disabled() is False
        monkeypatch.setattr(settings, "system_one_disabled", True)
        assert system_one_disabled() is True


class TestTheEgressCheck:
    def _phi_store(self):
        from continuum.security.policy import AccessPolicy, PolicyStore

        store = PolicyStore()
        store.add_policy(
            AccessPolicy(
                name="phi-stays-local",
                subjects=["phi"],
                resources=["system_one:remote:*"],
                effect="deny",
                denial_message="PHI may not leave the host.",
            )
        )
        return store

    async def test_a_tainted_run_is_denied_a_remote_backend(self, fake_classifier_cls):
        from continuum.security.policy_context import use_active_policy
        from continuum.system_one import SystemOneAccessDeniedError, classify

        class Ctx:
            data_labels = {"phi"}

        backend = fake_classifier_cls(egress="remote")
        with use_active_policy(self._phi_store(), "clinic", Ctx()):
            with pytest.raises(SystemOneAccessDeniedError):
                await classify("s", {"q": _binary()}, classifier=backend)
        assert backend.calls == []

    async def test_the_same_run_may_use_a_local_backend(self, fake_classifier_cls):
        from continuum.security.policy_context import use_active_policy
        from continuum.system_one import classify

        class Ctx:
            data_labels = {"phi"}

        backend = fake_classifier_cls(egress="local")
        with use_active_policy(self._phi_store(), "clinic", Ctx()):
            resp = await classify("s", {"q": _binary()}, classifier=backend)
        assert "q" in resp.answers

    async def test_the_denial_is_a_policy_outcome_not_a_failure(self):
        """Observability treats PolicyDeniedError as an expected governance result."""
        from continuum.exceptions import PolicyDeniedError
        from continuum.system_one import SystemOneAccessDeniedError

        assert issubclass(SystemOneAccessDeniedError, PolicyDeniedError)

    async def test_the_resource_names_egress_backend_and_model(self, fake_classifier_cls):
        from continuum.security.policy import AccessPolicy, PolicyStore
        from continuum.security.policy_context import use_active_policy
        from continuum.system_one import SystemOneAccessDeniedError, classify

        class Ctx:
            data_labels: set[str] = set()

        store = PolicyStore()
        store.add_policy(
            AccessPolicy(
                name="no-fake-7",
                subjects=["*"],
                resources=["system_one:local:fake:fake-7"],
                effect="deny",
            )
        )
        with use_active_policy(store, "agent", Ctx()):
            with pytest.raises(SystemOneAccessDeniedError):
                await classify(
                    "s", {"q": _binary()}, classifier=fake_classifier_cls(model="fake-7")
                )


class TestTypedAccessors:
    async def test_each_kind_has_an_accessor(self, fake_classifier_cls):
        from continuum.system_one import classify

        resp = await classify(
            "s", {"b": _binary(), "c": _choice(), "s": _score()}, classifier=fake_classifier_cls()
        )
        assert resp.binary("b").kind == "binary"
        assert resp.choice("c").kind == "choice"
        assert resp.score("s").kind == "score"

    async def test_asking_for_the_wrong_kind_is_an_error(self, fake_classifier_cls):
        from continuum.system_one import classify

        resp = await classify("s", {"b": _binary()}, classifier=fake_classifier_cls())
        with pytest.raises(TypeError):
            resp.choice("b")
