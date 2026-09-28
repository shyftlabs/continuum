"""The local NLI adapter, against an injected fake CrossEncoder.

A natural-language-inference model scores (premise, hypothesis) pairs as
contradiction / entailment / neutral. The adapter uses the state as the premise
and turns each question into hypotheses: a binary question's true-criterion (or
its instructions), a choice's label descriptions. It runs on this machine, so it
declares egress "local" -- the backend a PHI-tainted run can still use.

No model is downloaded here; the live run lives in tests/integration.
"""

from __future__ import annotations

import pytest

from continuum.system_one.testing import SystemOneContract


class FakeCrossEncoder:
    """Scores pairs from a script: hypothesis text -> (contradiction, entailment, neutral)."""

    def __init__(self, scores=None, *, id2label=None, fail=False):
        self._scores = scores or {}
        self.config = type(
            "Cfg", (), {"id2label": id2label or {0: "contradiction", 1: "entailment", 2: "neutral"}}
        )()
        self._fail = fail
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs, apply_softmax=False, **kwargs):
        if self._fail:
            raise RuntimeError("CUDA out of memory")
        assert apply_softmax, "the adapter must ask for probabilities, not logits"
        self.calls.append(list(pairs))
        order = [self.config.id2label[i] for i in range(3)]
        out = []
        for _premise, hypothesis in pairs:
            c, e, n = self._scores.get(hypothesis, (0.2, 0.5, 0.3))
            by_name = {"contradiction": c, "entailment": e, "neutral": n}
            out.append([by_name[name] for name in order])
        return out


def _local(ce=None, **kwargs):
    from continuum.system_one.backends.local_nli import LocalNLIClassifier

    return LocalNLIClassifier(cross_encoder=ce or FakeCrossEncoder(), **kwargs)


class TestHypotheses:
    async def test_a_binary_question_uses_its_true_criterion_as_the_hypothesis(self):
        """NLI judges statements, not questions: 'The task is complete.' is
        something a premise can entail; 'Is the task complete?' is not."""
        from continuum.system_one import BinaryQuestion

        ce = FakeCrossEncoder()
        await _local(ce).classify(
            "Here is the final report.",
            {"done": BinaryQuestion(instructions="Is it done?", true_criteria="The task is complete.")},
        )
        assert ce.calls[0] == [("Here is the final report.", "The task is complete.")]

    async def test_without_a_true_criterion_the_instructions_are_used(self):
        from continuum.system_one import BinaryQuestion

        ce = FakeCrossEncoder()
        await _local(ce).classify("text", {"q": BinaryQuestion(instructions="The text is spam.")})
        assert ce.calls[0] == [("text", "The text is spam.")]

    async def test_a_choice_asks_one_hypothesis_per_label_description(self):
        from continuum.system_one import ChoiceQuestion

        ce = FakeCrossEncoder()
        await _local(ce).classify(
            "My card was charged twice.",
            {
                "team": ChoiceQuestion(
                    instructions="Which team?",
                    labels={"billing": "This is about a payment.", "technical": None},
                )
            },
        )
        hypotheses = [h for _p, h in ce.calls[0]]
        assert hypotheses == ["This is about a payment.", "technical"]

    async def test_all_pairs_go_in_one_batch(self):
        from continuum.system_one import BinaryQuestion, ChoiceQuestion

        ce = FakeCrossEncoder()
        await _local(ce).classify(
            "s",
            {
                "a": BinaryQuestion(instructions="A holds."),
                "b": ChoiceQuestion(instructions="Pick", labels={"x": "X holds.", "y": "Y holds."}),
            },
        )
        assert len(ce.calls) == 1 and len(ce.calls[0]) == 3


class TestProbabilities:
    async def test_binary_is_entailment_against_contradiction(self):
        """Neutral is dropped: P(true) = e / (e + c)."""
        from continuum.system_one import BinaryQuestion

        ce = FakeCrossEncoder({"It is done.": (0.1, 0.6, 0.3)})
        raw = await _local(ce).classify("s", {"q": BinaryQuestion(instructions="It is done.")})
        assert raw.distributions["q"]["true"] == pytest.approx(0.6 / 0.7)
        assert raw.distributions["q"]["false"] == pytest.approx(0.1 / 0.7)

    async def test_choice_normalises_entailment_across_labels(self):
        from continuum.system_one import ChoiceQuestion

        ce = FakeCrossEncoder({"X holds.": (0.1, 0.6, 0.3), "Y holds.": (0.5, 0.2, 0.3)})
        raw = await _local(ce).classify(
            "s", {"q": ChoiceQuestion(instructions="Pick", labels={"x": "X holds.", "y": "Y holds."})}
        )
        assert raw.distributions["q"] == {"x": pytest.approx(0.75), "y": pytest.approx(0.25)}

    async def test_the_models_own_label_order_is_respected(self):
        """Models differ in where 'entailment' sits; hard-coding index 1 would
        silently read contradiction as agreement on some of them."""
        from continuum.system_one import BinaryQuestion

        ce = FakeCrossEncoder(
            {"It is done.": (0.1, 0.6, 0.3)},
            id2label={0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"},
        )
        raw = await _local(ce).classify("s", {"q": BinaryQuestion(instructions="It is done.")})
        assert raw.distributions["q"]["true"] == pytest.approx(0.6 / 0.7)


class TestFailuresAndLoading:
    async def test_a_model_error_is_a_backend_error(self):
        from continuum.system_one import BinaryQuestion, SystemOneBackendError

        with pytest.raises(SystemOneBackendError):
            await _local(FakeCrossEncoder(fail=True)).classify(
                "s", {"q": BinaryQuestion(instructions="x.")}
            )

    async def test_a_model_without_entailment_labels_is_refused(self):
        from continuum.system_one import BinaryQuestion, SystemOneNotConfiguredError

        ce = FakeCrossEncoder(id2label={0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})
        with pytest.raises(SystemOneNotConfiguredError):
            await _local(ce).classify("s", {"q": BinaryQuestion(instructions="x.")})

    async def test_missing_sentence_transformers_names_the_extra(self, monkeypatch):
        import builtins

        from continuum.system_one import BinaryQuestion, SystemOneNotConfiguredError
        from continuum.system_one.backends.local_nli import LocalNLIClassifier

        real_import = builtins.__import__

        def no_st(name, *args, **kwargs):
            if name.startswith("sentence_transformers"):
                raise ImportError("No module named 'sentence_transformers'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_st)
        backend = LocalNLIClassifier(model="cross-encoder/nli-deberta-v3-small")
        with pytest.raises(SystemOneNotConfiguredError) as exc:
            await backend.classify("s", {"q": BinaryQuestion(instructions="x.")})
        assert "embeddings" in str(exc.value)

    async def test_the_model_is_loaded_once(self, monkeypatch):
        from continuum.system_one import BinaryQuestion
        from continuum.system_one.backends import local_nli

        loads = []

        def fake_load(self):
            loads.append(self.model)
            return FakeCrossEncoder()

        monkeypatch.setattr(local_nli.LocalNLIClassifier, "_load", fake_load)
        backend = local_nli.LocalNLIClassifier(model="m")
        for _ in range(3):
            await backend.classify("s", {"q": BinaryQuestion(instructions="x.")})
        assert loads == ["m"]

    def test_constructing_it_loads_nothing(self, monkeypatch):
        """The registry builds backends lazily, and agents may be built at import."""
        from continuum.system_one.backends import local_nli

        monkeypatch.setattr(
            local_nli.LocalNLIClassifier,
            "_load",
            lambda self: pytest.fail("model loaded at construction"),
        )
        local_nli.LocalNLIClassifier(model="m")

    def test_it_declares_itself_local_and_text_only(self):
        caps = _local().capabilities
        assert caps.egress == "local"
        assert caps.structured_state is False
        assert caps.question_types == frozenset({"binary", "choice"})

    def test_the_registry_builds_it_from_a_spec(self):
        from continuum.system_one import create_classifier

        backend = create_classifier("local:cross-encoder/nli-deberta-v3-small")
        assert (backend.name, backend.model) == ("local", "cross-encoder/nli-deberta-v3-small")


class TestLocalNLIMeetsTheContract(SystemOneContract):
    latency_budget_ms = 200.0  # fake model: this checks the adapter's own overhead

    def make_classifier(self):
        return _local()

    def make_failing_classifier(self):
        return _local(FakeCrossEncoder(fail=True))
