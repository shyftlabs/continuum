"""
Workflow agents for multi-agent orchestration.

Provides specialized agents for workflow patterns:
- RouterAgent: Dynamic routing/triage
- SequentialAgent: Pipeline execution
- ParallelAgent: Concurrent execution (same input to all agents)
- LoopAgent: Iterative execution
- SupervisedSequentialAgent: Sequential with LLM quality gating per step
- ScatterAgent: Scatter/gather — LLM splits input into slices, each agent gets its own
- DebateAgent: Pro + con + judge synthesis pattern
"""

from orchestrator.agent.workflow.debate import DebateAgent, create_debate_agent
from orchestrator.agent.workflow.loop import LoopAgent
from orchestrator.agent.workflow.parallel import ParallelAgent
from orchestrator.agent.workflow.planner import PlannerAgent, create_planner_agent
from orchestrator.agent.workflow.reflection import ReflectionAgent, create_reflection_agent, generate_critique_prompt
from orchestrator.agent.workflow.router import RouterAgent
from orchestrator.agent.workflow.sequential import SequentialAgent, create_sequential_agent
from orchestrator.agent.workflow.scatter import ScatterAgent, create_scatter_agent
from orchestrator.agent.workflow.supervised import SupervisedSequentialAgent, create_supervised_agent

__all__ = [
    "RouterAgent",
    "SequentialAgent",
    "create_sequential_agent",
    "ParallelAgent",
    "LoopAgent",
    "ReflectionAgent",
    "create_reflection_agent",
    "generate_critique_prompt",
    "PlannerAgent",
    "create_planner_agent",
    "SupervisedSequentialAgent",
    "create_supervised_agent",
    "ScatterAgent",
    "create_scatter_agent",
    "DebateAgent",
    "create_debate_agent",
]
