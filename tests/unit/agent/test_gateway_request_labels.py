"""Model calls through the Smart Gateway carry the caller's labels.

The gateway attributes cost and applies budgets from body.metadata. Continuum
sent only session_id and trace_id, so a product could not ask the gateway for
cost by tenant, fund or feature; and building that metadata replaced the
agent's own extra_body outright.
"""

from __future__ import annotations

import pytest

from continuum.agent.execution import executor
from continuum.agent.execution.executor import _enrich_config_for_gateway
from continuum.agent.types import RunContext
from continuum.llm.config import LLMConfig


@pytest.fixture
def gateway(monkeypatch):
    monkeypatch.setattr(executor.settings, "smart_gateway_url", "https://gateway.test/v1")


def test_caller_labels_and_tags_are_sent(gateway):
    ctx = RunContext(
        run_id="run-1",
        session_id="s-1",
        trace_id="t-1",
        tags=["ask-arnie"],
        metadata={"gateway_metadata": {"org_id": "org-7", "fund_id": "F-12", "skip": None}},
    )
    meta = _enrich_config_for_gateway(LLMConfig(model="auto"), ctx).extra_body["metadata"]
    assert meta == {
        "org_id": "org-7",
        "fund_id": "F-12",
        "tags": "ask-arnie",
        "session_id": "s-1",
        "trace_id": "t-1",
    }


def test_ids_cannot_be_overridden_by_labels(gateway):
    ctx = RunContext(run_id="run-1", metadata={"gateway_metadata": {"session_id": "spoof"}})
    meta = _enrich_config_for_gateway(LLMConfig(model="auto"), ctx).extra_body["metadata"]
    assert meta["session_id"] == "run-1"


def test_agent_extra_body_is_kept(gateway):
    cfg = LLMConfig(model="auto", extra_body={"top_k": 5, "metadata": {"team": "research"}})
    out = _enrich_config_for_gateway(cfg, RunContext(run_id="r", session_id="s"))
    assert out.extra_body["top_k"] == 5
    assert out.extra_body["metadata"]["team"] == "research"
    assert out.extra_body["metadata"]["session_id"] == "s"
    assert cfg.extra_body == {"top_k": 5, "metadata": {"team": "research"}}  # not mutated


def test_internal_run_metadata_is_not_sent(gateway):
    ctx = RunContext(run_id="r", metadata={"_filtered_tools": ["x"], "promoted_tools": {"y"}})
    meta = _enrich_config_for_gateway(LLMConfig(model="auto"), ctx).extra_body["metadata"]
    assert set(meta) == {"session_id", "trace_id"}


def test_unchanged_without_a_gateway(monkeypatch):
    monkeypatch.setattr(executor.settings, "smart_gateway_url", None)
    cfg = LLMConfig(model="gpt-4o-mini")
    assert _enrich_config_for_gateway(cfg, RunContext(run_id="r")) is cfg
