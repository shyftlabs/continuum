"""
LeadFlow Temporal worker.

setup_registry() — async: connects FakeTwilioMCP, builds ToolExecutor,
                   creates all agents, registers them in the AgentRegistry,
                   sets a runner factory with all agents pre-loaded so
                   crm-lookup is available for voice-agent handoffs.

start_worker()   — starts the Temporal worker on the configured task queue.
stop_worker()    — graceful shutdown.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=True)

_root = Path(__file__).resolve().parents[2]
_hack = Path(__file__).resolve().parents[2]
for p in (_root / "src", _hack):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from agents.scoring import make_scoring_agent
from agents.scrapers import make_google_maps_agent, make_linkedin_agent, make_web_agent
from agents.voice import make_crm_lookup_agent, make_voice_agent
from config import LeadFlowConfig, default_config
from tools.mock_twilio import FakeTwilioMCP

from orchestrator import AgentRunner, RunnerConfig
from orchestrator.temporal.client import TemporalClient
from orchestrator.temporal.registry import AgentRegistry, get_agent_registry
from orchestrator.temporal.worker import WorkerManager

try:
    from orchestrator import MCPUtil, ToolExecutor
except ImportError:
    MCPUtil = None  # type: ignore
    ToolExecutor = None  # type: ignore


_worker_manager: WorkerManager | None = None


async def setup_registry(config: LeadFlowConfig | None = None) -> AgentRegistry:
    cfg = config or default_config
    m = cfg.model

    # Connect mock Twilio and build ToolExecutor for voice agent
    twilio_mcp = FakeTwilioMCP()
    await twilio_mcp.connect()
    voice_tools: list = []
    voice_executor = None
    if MCPUtil and ToolExecutor:
        tool_defs = await MCPUtil.get_function_tools(twilio_mcp)
        voice_tools = [t.model_dump() if hasattr(t, "model_dump") else t for t in tool_defs]
        voice_executor = ToolExecutor({twilio_mcp: None})
        await voice_executor.initialize()

    # Build all agents
    google_maps = make_google_maps_agent(m, cfg.leads_per_source)
    linkedin = make_linkedin_agent(m, cfg.leads_per_source)
    web = make_web_agent(m, cfg.leads_per_source)
    scoring = make_scoring_agent(m)
    crm_lookup = make_crm_lookup_agent(m)
    voice = make_voice_agent(m, voice_executor, voice_tools)

    all_agents = {a.name: a for a in [google_maps, linkedin, web, scoring, crm_lookup, voice]}

    # Use the global singleton — run_agent_activity calls get_agent_registry()
    # so agents must be in this exact instance, not a separate one.
    registry = get_agent_registry()
    registry.register_many(list(all_agents.values()))

    # Runner factory pre-loads all agents so crm-lookup is available for handoffs
    def _runner_factory():
        return AgentRunner(
            agent_registry=all_agents,
            config=RunnerConfig(persist_state=False, default_max_turns=cfg.max_turns),
        )

    registry.set_runner_factory(_runner_factory)
    return registry


async def start_worker(
    client: TemporalClient,
    registry: AgentRegistry,
    config: LeadFlowConfig | None = None,
) -> WorkerManager:
    global _worker_manager
    cfg = config or default_config
    manager = WorkerManager(client=client, registry=registry)
    await manager.start(task_queue=cfg.task_queue)
    _worker_manager = manager
    return manager


async def stop_worker() -> None:
    global _worker_manager
    if _worker_manager:
        await _worker_manager.stop()
        _worker_manager = None
