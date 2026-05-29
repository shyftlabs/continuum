"""
Tests for RouterAgent._stamp_priority — ensures RunContext.priority is set
from the matched route's dispatch_priority after routing.
"""
from __future__ import annotations

import pytest

from orchestrator.agent.types import Route, RunContext
from orchestrator.agent.workflow.router import RouterAgent


def _make_context():
    return RunContext(run_id="test-run")


def _make_router(routes: list[Route]) -> RouterAgent:
    return RouterAgent(name="triage", routes=routes)


class TestStampPriority:
    def test_stamps_priority_from_matched_route(self):
        route = Route(agent_name="premium_agent", description="Premium", dispatch_priority=9)
        router = _make_router([route])
        ctx = _make_context()

        router._stamp_priority("premium_agent", ctx)

        assert ctx.priority == 9

    def test_stamps_default_priority_when_not_overridden(self):
        route = Route(agent_name="standard_agent", description="Standard")
        router = _make_router([route])
        ctx = _make_context()

        router._stamp_priority("standard_agent", ctx)

        assert ctx.priority == 5  # Route.dispatch_priority default

    def test_no_op_when_context_is_none(self):
        route = Route(agent_name="agent", description="desc", dispatch_priority=8)
        router = _make_router([route])

        # Should not raise
        router._stamp_priority("agent", None)

    def test_no_op_when_agent_name_is_none(self):
        route = Route(agent_name="agent", description="desc", dispatch_priority=8)
        router = _make_router([route])
        ctx = _make_context()
        original_priority = ctx.priority

        router._stamp_priority(None, ctx)

        assert ctx.priority == original_priority

    def test_no_op_when_route_not_found(self):
        route = Route(agent_name="agent_a", description="desc", dispatch_priority=8)
        router = _make_router([route])
        ctx = _make_context()
        original_priority = ctx.priority

        router._stamp_priority("unknown_agent", ctx)

        assert ctx.priority == original_priority

    def test_different_routes_have_different_priorities(self):
        routes = [
            Route(agent_name="urgent", description="Urgent", dispatch_priority=10),
            Route(agent_name="batch", description="Batch", dispatch_priority=1),
        ]
        router = _make_router(routes)

        ctx_urgent = _make_context()
        router._stamp_priority("urgent", ctx_urgent)
        assert ctx_urgent.priority == 10

        ctx_batch = _make_context()
        router._stamp_priority("batch", ctx_batch)
        assert ctx_batch.priority == 1


class TestRouteDispatchPriorityDefault:
    def test_route_dispatch_priority_defaults_to_5(self):
        route = Route(agent_name="agent", description="desc")
        assert route.dispatch_priority == 5

    def test_route_dispatch_priority_settable(self):
        route = Route(agent_name="agent", description="desc", dispatch_priority=7)
        assert route.dispatch_priority == 7
