"""The Laya backend: Convai's open-weight typed-decision model, run locally.

``laya:<checkpoint>`` runs upstream ``laya`` (PyPI, Apache-2.0 code and weights;
CPU/CUDA/MPS/XPU). ``laya-mlx:<checkpoint>`` runs the independent MLX port on
Apple Silicon -- same API and answer format, measured to agree with upstream to
within 0.0018 on the same checkpoint.

Laya speaks the typed-decision wire format Jev does, so these tests check what
is specific to it: it is a local, in-process model (loaded lazily, once, run
off the event loop), it reports its own model name ("laya-rl-agent") where the
checkpoint id is what a trace needs, and it truncates long state silently --
which the adapter detects from usage and says so.

The fake mirrors upstream as measured: usage.input_tokens is summed over one row
per question, each capped at cfg["max_len"] (512 for the English checkpoint).
"""

from __future__ import annotations

import logging
import threading

import pytest

from continuum.system_one.testing import SystemOneContract


class FakeLayaAgent:
    def __init__(self, *, fail=None, tokens_per_row=36, max_len=512, drop=False):
        self.cfg = {"max_len": max_len}
        self.calls: list[tuple] = []
        self.threads: list[str] = []
        self._fail = fail
        self._tokens = tokens_per_row
        self._drop = drop

    def predict(self, state, questions):
        self.threads.append(threading.current_thread().name)
        self.calls.append((state, questions))
        if self._fail is not None:
            raise self._fail
        answers = {}
        for qid, q in questions.items():
            if q["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": 0.2, "confidence": 0.8}
            elif q["type"] == "choice":
                labels = list(q["criteria"])
                rest = round(0.4 / (len(labels) - 1), 4)
                answers[qid] = {
                    "type": "choice",
                    "choice": labels[0],
                    "probabilities": {
                        label: (0.6 if i == 0 else rest) for i, label in enumerate(labels)
                    },
                    "confidence": 0.41,
                }
            else:
                n = len(q["criteria"])
                answers[qid] = {
                    "type": "score",
                    "score": (n - 1) / 2,
                    "probabilities": {str(i): round(1.0 / n, 4) for i in range(n)},
                    "confidence": 0.0,
                }
        if self._drop:
            answers.clear()
        rows = len(questions)
        return {
            "model": "laya-rl-agent",
            "answers": answers,
            "usage": {
                "input_tokens": min(self._tokens, self.cfg["max_len"]) * rows,
                "output_tokens": 0,
            },
        }


def _laya(agent=None, **kwargs):
    from continuum.system_one.backends.laya import LayaClassifier

    return LayaClassifier(agent=agent or FakeLayaAgent(), **kwargs)


def _q():
    from continuum.system_one import BinaryQuestion, ChoiceQuestion, ScoreQuestion

    return {
        "risky": BinaryQuestion(
            instructions="Would executing `tool_call` be risky?",
            true_criteria="It could cause harm.",
            false_criteria="It is low risk.",
        ),
        "team": ChoiceQuestion(
            instructions="Which team?", labels={"billing": "Payments.", "technical": "Bugs."}
        ),
        "severity": ScoreQuestion(instructions="How severe?", levels=["low", "mid", "high"]),
    }


class TestTranslation:
    async def test_questions_go_out_in_the_typed_decision_format(self):
        agent = FakeLayaAgent()
        await _laya(agent).classify({"msg": "hi"}, _q())

        state, questions = agent.calls[0]
        assert state == {"msg": "hi"}, "Laya accepts JSON state; it must not be stringified"
        assert questions["risky"] == {
            "type": "noul",
            "instructions": "Would executing `tool_call` be risky?",
            "criteria": {"true": "It could cause harm.", "false": "It is low risk."},
        }
        assert questions["team"]["criteria"] == {"billing": "Payments.", "technical": "Bugs."}
        assert questions["severity"]["criteria"] == ["low", "mid", "high"]

    async def test_answers_become_raw_distributions(self):
        raw = await _laya().classify("s", _q())
        assert raw.distributions["risky"] == {
            "true": pytest.approx(0.2),
            "false": pytest.approx(0.8),
        }
        assert raw.distributions["team"] == {
            "billing": pytest.approx(0.6),
            "technical": pytest.approx(0.4),
        }
        assert set(raw.distributions["severity"]) == {0, 1, 2}
        assert raw.raw_confidence["team"] == pytest.approx(0.41)

    async def test_the_checkpoint_id_is_reported_not_laya_rl_agent(self):
        """Laya names every checkpoint 'laya-rl-agent'; a trace needs to know
        which checkpoint decided."""
        raw = await _laya(model="convaiinnovations/laya-typed-decisions").classify("s", _q())
        assert raw.model == "convaiinnovations/laya-typed-decisions"

    async def test_a_missing_answer_is_a_response_error(self):
        from continuum.system_one import SystemOneResponseError

        with pytest.raises(SystemOneResponseError):
            await _laya(FakeLayaAgent(drop=True)).classify("s", _q())


class TestTruncation:
    """Upstream truncates long state silently (right for text/JSON, left for
    conversation lists). Detected here from usage: every row at max_len means
    the state did not fit. A row cut while shorter-question rows fit is not
    detected -- the heuristic under-reports, it never invents truncation."""

    def _collect(self):
        messages: list[str] = []

        class Collector(logging.Handler):
            def emit(self, record):
                messages.append(record.getMessage())

        handler = Collector()
        logging.getLogger("continuum.system_one.backends.laya").addHandler(handler)
        return messages, handler

    async def test_state_that_did_not_fit_is_flagged_and_logged(self):
        messages, h = self._collect()
        try:
            raw = await _laya(FakeLayaAgent(tokens_per_row=10_000)).classify(
                "SECRET-PAYLOAD " * 3000, _q()
            )
        finally:
            logging.getLogger("continuum.system_one.backends.laya").removeHandler(h)
        assert raw.usage["truncated"] is True
        assert any("truncated" in m and "512" in m for m in messages)
        assert not any("SECRET-PAYLOAD" in m for m in messages), "never log the state"

    async def test_state_that_fit_is_not_flagged(self):
        raw = await _laya(FakeLayaAgent(tokens_per_row=36)).classify("short", _q())
        assert "truncated" not in raw.usage
        assert raw.usage["input_tokens"] == 108


class TestFailuresAndLoading:
    async def test_a_model_error_is_a_backend_error(self):
        from continuum.system_one import SystemOneBackendError

        with pytest.raises(SystemOneBackendError):
            await _laya(FakeLayaAgent(fail=RuntimeError("MPS out of memory"))).classify("s", _q())

    async def test_a_rejected_question_is_a_backend_error_too(self):
        """Laya raises ValueError for questions it cannot fit (e.g. options over
        head_max_len); that must not escape as a raw ValueError."""
        from continuum.system_one import SystemOneBackendError

        with pytest.raises(SystemOneBackendError):
            await _laya(FakeLayaAgent(fail=ValueError("options exceed head_max_len"))).classify(
                "s", _q()
            )

    async def test_prediction_runs_off_the_event_loop(self):
        agent = FakeLayaAgent()
        await _laya(agent).classify("s", _q())
        assert agent.threads and agent.threads[0] != threading.main_thread().name

    async def test_missing_laya_names_the_extra(self, monkeypatch):
        import builtins

        from continuum.system_one import SystemOneNotConfiguredError
        from continuum.system_one.backends.laya import LayaClassifier

        real_import = builtins.__import__

        def no_laya(name, *args, **kwargs):
            if name == "laya" or name.startswith("laya."):
                raise ImportError("No module named 'laya'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_laya)
        with pytest.raises(SystemOneNotConfiguredError) as exc:
            await LayaClassifier().classify("s", _q())
        assert "[laya]" in str(exc.value)

    async def test_the_model_is_loaded_once_with_the_device(self, monkeypatch):
        from continuum.system_one.backends import laya as laya_backend

        loads = []

        def fake_load(self):
            loads.append((self.model, self._device))
            return FakeLayaAgent()

        monkeypatch.setattr(laya_backend.LayaClassifier, "_load", fake_load)
        backend = laya_backend.LayaClassifier(model="m", device="cpu")
        for _ in range(3):
            await backend.classify("s", _q())
        assert loads == [("m", "cpu")]

    def test_constructing_it_loads_nothing(self, monkeypatch):
        from continuum.system_one.backends import laya as laya_backend

        monkeypatch.setattr(
            laya_backend.LayaClassifier,
            "_load",
            lambda self: pytest.fail("model loaded at construction"),
        )
        laya_backend.LayaClassifier()

    def test_it_declares_itself_local_structured_and_complete(self):
        caps = _laya().capabilities
        assert caps.egress == "local"
        assert caps.structured_state is True
        assert caps.question_types == frozenset({"binary", "choice", "score"})

    def test_the_default_checkpoint_is_the_english_base(self):
        """Measured best of the three checkpoints on Continuum's own seam
        questions; the fine-tuned typed-decisions one separated risky from safe
        calls less well (0.44/0.50 against 0.32/0.54)."""
        from continuum.system_one.backends.laya import DEFAULT_MODEL

        assert DEFAULT_MODEL == "convaiinnovations/laya"


class TestTheRegistry:
    def test_laya_builds_the_upstream_backend(self):
        from continuum.system_one import create_classifier

        backend = create_classifier("laya:convaiinnovations/laya")
        assert (backend.name, backend.model) == ("laya", "convaiinnovations/laya")

    def test_laya_mlx_builds_the_mlx_backend(self):
        from continuum.system_one import create_classifier
        from continuum.system_one.backends.laya import LayaMLXClassifier

        backend = create_classifier("laya-mlx:aac6fef/laya-mlx")
        assert isinstance(backend, LayaMLXClassifier)
        assert (backend.name, backend.model) == ("laya-mlx", "aac6fef/laya-mlx")

    async def test_missing_laya_mlx_names_its_extra(self, monkeypatch):
        import builtins

        from continuum.system_one import SystemOneNotConfiguredError
        from continuum.system_one.backends.laya import LayaMLXClassifier

        real_import = builtins.__import__

        def no_mlx(name, *args, **kwargs):
            if name == "laya_mlx" or name.startswith("laya_mlx."):
                raise ImportError("No module named 'laya_mlx'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_mlx)
        with pytest.raises(SystemOneNotConfiguredError) as exc:
            await LayaMLXClassifier(model="aac6fef/laya-mlx").classify("s", _q())
        assert "[laya-mlx]" in str(exc.value)
        assert "Apple Silicon" in str(exc.value)


class TestThePackagingExtrasExist:
    def test_both_extras_are_declared(self):
        import tomllib
        from pathlib import Path

        pyproject = tomllib.loads(
            (Path(__file__).resolve().parents[3] / "pyproject.toml").read_text()
        )
        extras = pyproject["project"]["optional-dependencies"]
        assert any(dep.startswith("laya>=") for dep in extras["laya"])
        (mlx,) = extras["laya-mlx"]
        assert mlx.startswith("laya-mlx>=")
        assert "platform_machine == 'arm64'" in mlx, "MLX only installs on Apple Silicon"


class TestLayaMeetsTheContract(SystemOneContract):
    latency_budget_ms = 200.0  # fake agent: this checks the adapter's own overhead

    def make_classifier(self):
        return _laya()

    def make_failing_classifier(self):
        return _laya(FakeLayaAgent(fail=RuntimeError("model error")))


class TestLayaMLXMeetsTheContract(SystemOneContract):
    latency_budget_ms = 200.0

    def make_classifier(self):
        from continuum.system_one.backends.laya import LayaMLXClassifier

        return LayaMLXClassifier(agent=FakeLayaAgent())

    def make_failing_classifier(self):
        from continuum.system_one.backends.laya import LayaMLXClassifier

        return LayaMLXClassifier(agent=FakeLayaAgent(fail=RuntimeError("model error")))
