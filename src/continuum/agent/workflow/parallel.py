"""
Parallel Agent - Concurrent execution agent.

Executes multiple agents concurrently and merges their results.

NOTE: Workflow agents now include Langfuse span tracing for full observability.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from continuum.agent.base import BaseAgent
from continuum.agent.config import ParallelConfig
from continuum.agent.exceptions import ParallelWorkflowError
from continuum.agent.types import (
    AgentResponse,
    FailStrategy,
    MergeStrategy,
    ResponseStatus,
    RunContext,
    TokenUsage,
)
from continuum.agent.utils.context_utils import publish_active_policy
from continuum.agent.workflow._forkable import (
    branch_outputs_from_trace,
    branch_recorder_context,
    link_lineage,
    resumed_input,
    segment_by_markers,
)
from continuum.logging import get_logger

if TYPE_CHECKING:
    from continuum.agent.runner import AgentRunner
    from continuum.llm import LLMClient

logger = get_logger(__name__)


@dataclass
class ParallelAgent(BaseAgent):
    """
    Executes multiple agents concurrently.

    All agents receive the same input and run in parallel.
    Results are merged according to the configured strategy.

    Merge strategies:
    - CONCATENATE: Simple concatenation of outputs
    - LLM_SUMMARIZE: LLM summarizes all outputs
    - STRUCTURED: Returns dict mapping agent names to outputs
    - FIRST_SUCCESS: Returns first successful result

    Example:
        ```python
        from continuum.agent import BaseAgent
        from continuum.agent.workflow import ParallelAgent
        from continuum.agent.types import MergeStrategy

        # Define parallel workers
        web_searcher = BaseAgent(
            name="web-search",
            instructions="Search the web for information.",
        )

        db_searcher = BaseAgent(
            name="db-search",
            instructions="Search internal database.",
        )

        # Create parallel executor
        parallel = ParallelAgent(
            name="parallel-search",
            agents=[web_searcher, db_searcher],
            merge_strategy=MergeStrategy.LLM_SUMMARIZE,
        )

        # Run parallel agents
        result = await runner.run(parallel, "Find information about X")
        ```
    """

    # Agents to execute in parallel
    agents: list[BaseAgent] = field(default_factory=list)

    # Parallel configuration
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)

    # Agent whose memory_config governs post-execution long-term memory writes.
    # If None (default), no memory is written after the parallel run completes.
    memory_agent: BaseAgent | None = None

    def __post_init__(self) -> None:
        """Initialize parallel agent."""
        if not self.name:
            from continuum.agent.exceptions import AgentConfigurationError

            raise AgentConfigurationError("Agent name is required")

        if not self.agents:
            from continuum.agent.exceptions import AgentConfigurationError

            raise AgentConfigurationError("ParallelAgent requires at least one agent")

    @publish_active_policy
    async def execute(
        self,
        input_text: str,
        runner: AgentRunner,
        context: RunContext,
        llm_client: LLMClient | None = None,
    ) -> AgentResponse:
        """
        Execute agents in parallel.

        Args:
            input_text: Input for all agents
            runner: Agent runner for executing sub-agents
            context: Run context
            llm_client: LLM client for summarization

        Returns:
            Merged AgentResponse
        """
        # Own the one decision trace spanning all branches (rooted at this agent).
        created = runner.ensure_recorder(context, self.name, input_text)
        context.suppress_session_log = True
        # ORDERED CAPTURE: give each branch its OWN recorder so concurrent runs
        # don't interleave into the shared trace; merge them back in order after.
        tasks: list[tuple[BaseAgent, Any, asyncio.Task[AgentResponse]]] = []
        for i, agent in enumerate(self.agents):
            branch_ctx, branch_rec = branch_recorder_context(context, index=i)
            task = asyncio.create_task(self._run_agent_safe(agent, input_text, runner, branch_ctx))
            tasks.append((agent, branch_rec, task))

        # Wait for all tasks with timeout
        results: dict[str, AgentResponse | Exception] = {}

        try:
            done, pending = await asyncio.wait(
                [t for _, _, t in tasks],
                timeout=self.parallel_config.timeout,
                return_when=asyncio.ALL_COMPLETED,
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()

            # Collect results
            for agent, _branch_rec, task in tasks:
                if task.done():
                    try:
                        results[agent.name] = task.result()
                    except Exception as e:
                        results[agent.name] = e
                else:
                    results[agent.name] = TimeoutError("Task timed out")

        except Exception as e:
            logger.error(f"Parallel execution failed: {e}")
            raise ParallelWorkflowError(
                f"Parallel execution failed: {e}",
                run_id=context.run_id,
                original_error=e,
            ) from e

        # Process results
        successful: dict[str, AgentResponse] = {}
        failed: dict[str, str] = {}

        for agent_name, result in results.items():
            if isinstance(result, AgentResponse):
                successful[agent_name] = result
            else:
                failed[agent_name] = str(result)

        # ORDERED CAPTURE: merge each successful branch's steps back into the owned
        # trace in agent order, so the concurrent branches land as deterministic,
        # contiguous, stage-indexed segments regardless of wall-clock interleaving.
        if context.recorder is not None:
            for i, (agent, branch_rec, _task) in enumerate(tasks):
                if branch_rec is not None:
                    # Absorb EVERY branch (incl. failed → empty/partial steps) so
                    # stage indices stay contiguous and the trace shows all
                    # branches, matching Scatter.
                    context.recorder.absorb(
                        branch_rec.trace.steps,
                        stage=i,
                        label=agent.name,
                        orchestrator_name=self.name,
                    )

        # Handle failures based on strategy
        if failed:
            if self.parallel_config.fail_strategy == FailStrategy.REQUIRE_ALL:
                raise ParallelWorkflowError(
                    f"Some agents failed: {list(failed.keys())}",
                    failed_agents=list(failed.keys()),
                    run_id=context.run_id,
                )
            elif self.parallel_config.fail_strategy == FailStrategy.FAIL_FAST and not successful:
                raise ParallelWorkflowError(
                    "All agents failed",
                    failed_agents=list(failed.keys()),
                    run_id=context.run_id,
                )

        if not successful:
            return AgentResponse(
                content="All parallel agents failed",
                agent_name=self.name,
                status=ResponseStatus.ERROR,
                error="; ".join(f"{k}: {v}" for k, v in failed.items()),
            )

        # Merge results
        merged = await self._merge_results(
            successful,
            input_text,
            llm_client,
        )

        # Record the synthesis/merge as its own stage so the final answer (and the
        # fact that a merge happened) is visible in the trace, mirroring Scatter's
        # explicit gather stage.
        if context.recorder is not None:
            context.recorder.record_workflow_step(
                self.name, stage=len(self.agents), label="merge", agent_stack=[self.name]
            )
            context.recorder.record_llm_call(
                self.name, 0, output=merged, decision="merge", agent_stack=[self.name]
            )

        # Calculate totals
        total_usage = TokenUsage()
        for resp in successful.values():
            total_usage = total_usage.add(resp.usage)

        result = AgentResponse(
            content=merged,
            agent_name=self.name,
            status=ResponseStatus.SUCCESS,
            usage=total_usage,
            agents_used=list(successful.keys()),
        )
        result.run_id = context.run_id

        if context.session_id:
            await runner.save_turn(
                session_id=context.session_id,
                user_message=input_text,
                assistant_message=merged,
                agent=self.memory_agent,
            )

        if created:
            await runner.persist_decision_trace(context, result)
        return result

    async def resume_from(
        self,
        *,
        parent_trace: Any,
        from_step: str,
        override: dict[str, Any] | None,
        runner: AgentRunner,
        context: RunContext,
    ) -> AgentResponse:
        """Re-run only the branch that owned ``from_step``, then re-merge.

        Forking a parallel run answers "what if branch X had a different input?".
        The other branches are not re-run — their cached outputs are recovered
        from the parent trace and merged with the re-run branch using the same
        merge strategy.
        """
        step_stage, stage_first = segment_by_markers(parent_trace)
        stage_idx = step_stage.get(from_step)
        if stage_idx is None:
            raise ValueError(f"resume_from: step '{from_step}' not found in trace")
        if stage_idx < 0 or stage_idx >= len(self.agents):
            raise ValueError(f"resume_from: branch index {stage_idx} out of range")

        branch_input = resumed_input(stage_first.get(stage_idx), override, parent_trace.user_query)

        created = runner.ensure_recorder(context, self.name, parent_trace.user_query)
        if created:
            link_lineage(context, parent_trace, from_step, override, stage_idx)
        context.suppress_session_log = True

        # Re-run only the forked branch, capturing it into its own recorder.
        agent = self.agents[stage_idx]
        branch_ctx, branch_rec = branch_recorder_context(context, index=stage_idx)
        rerun = await self._run_agent_safe(agent, branch_input, runner, branch_ctx)
        if context.recorder is not None and branch_rec is not None:
            context.recorder.absorb(
                branch_rec.trace.steps,
                stage=stage_idx,
                label=agent.name,
                orchestrator_name=self.name,
            )

        # Recover the OTHER branches' cached outputs and replace the forked one.
        siblings = branch_outputs_from_trace(parent_trace)
        siblings[stage_idx] = rerun.content or ""

        # Rebuild a {agent_name: AgentResponse} map in agent order for merging.
        merged_inputs: dict[str, AgentResponse] = {}
        for i, a in enumerate(self.agents):
            if i not in siblings:
                continue
            if i == stage_idx:
                merged_inputs[a.name] = rerun
            else:
                merged_inputs[a.name] = AgentResponse(
                    content=siblings[i],
                    agent_name=a.name,
                    status=ResponseStatus.SUCCESS,
                )

        merged = await self._merge_results(merged_inputs, branch_input, None)

        total_usage = TokenUsage()
        for resp in merged_inputs.values():
            total_usage = total_usage.add(resp.usage)

        result = AgentResponse(
            content=merged,
            agent_name=self.name,
            status=ResponseStatus.SUCCESS,
            usage=total_usage,
            agents_used=list(merged_inputs.keys()),
        )
        result.run_id = context.run_id

        if created:
            await runner.persist_decision_trace(context, result)
        return result

    async def _run_agent_safe(
        self,
        agent: BaseAgent,
        input_text: str,
        runner: AgentRunner,
        context: RunContext,
    ) -> AgentResponse:
        """Run an agent with error handling."""
        try:
            return await runner.run(
                agent=agent,
                input=input_text,
                context=context,
            )
        except Exception as e:
            logger.error(f"Agent {agent.name} failed: {e}")
            raise

    async def _merge_results(
        self,
        results: dict[str, AgentResponse],
        input_text: str,
        llm_client: LLMClient | None,
    ) -> str:
        """Merge results from multiple agents."""
        strategy = self.parallel_config.merge_strategy

        if strategy == MergeStrategy.FIRST_SUCCESS:
            # Return first result
            return next(iter(results.values())).content

        elif strategy == MergeStrategy.CONCATENATE:
            return self._concatenate_results(results)

        elif strategy == MergeStrategy.STRUCTURED:
            # Return as structured text
            import json

            return json.dumps({name: resp.content for name, resp in results.items()}, indent=2)

        elif strategy == MergeStrategy.LLM_SUMMARIZE:
            # Use LLM to summarize
            if llm_client is None:
                from continuum.core.container import get_container

                llm_client = get_container().llm_client

            # Build prompt
            outputs = "\n\n".join(f"### {name}\n{resp.content}" for name, resp in results.items())

            if self.parallel_config.summary_prompt:
                prompt = f"{outputs}\n\n{self.parallel_config.summary_prompt}"
            else:
                prompt = f"""Multiple agents were asked to address the following request:

Request: {input_text}

Here are their responses:

{outputs}

Please synthesize these responses into a single coherent answer that captures the key information from all sources."""

            logger.info(
                "===== FINAL PROMPT [%s/merge] =====\n[user] %s\n========================",
                self.name,
                prompt,
            )
            try:
                from continuum.llm.config import LLMConfig

                response = await llm_client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    config=LLMConfig(
                        model=self.parallel_config.summary_model or self.model,
                        temperature=self.parallel_config.summary_temperature,
                    ),
                )
                return response.content
            except Exception as e:
                logger.warning(f"LLM merge failed: {e}, falling back to concatenation")
                return self._concatenate_results(results)

        # Default: concatenate
        return self._concatenate_results(results)

    def _concatenate_results(self, results: dict[str, AgentResponse]) -> str:
        """Simple concatenation of results."""
        parts = []
        for agent_name, response in results.items():
            parts.append(f"## {agent_name}\n{response.content}")
        return "\n\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update(
            {
                "agents": [a.name for a in self.agents],
                "parallel_config": self.parallel_config.to_dict(),
                "workflow_type": "parallel",
            }
        )
        return base


def create_parallel_agent(
    name: str,
    agents: list[BaseAgent],
    *,
    merge_strategy: MergeStrategy = MergeStrategy.LLM_SUMMARIZE,
    fail_strategy: FailStrategy = FailStrategy.CONTINUE_ON_ERROR,
    timeout: int = 300,
    memory_agent: BaseAgent | None = None,
) -> ParallelAgent:
    """
    Factory function to create a parallel agent.

    Args:
        name: Agent name
        agents: List of agents to execute in parallel
        merge_strategy: How to merge results
        fail_strategy: How to handle failures
        timeout: Timeout in seconds
        memory_agent: Agent whose memory_config governs post-execution long-term memory writes

    Returns:
        Configured ParallelAgent
    """
    return ParallelAgent(
        name=name,
        agents=agents,
        memory_agent=memory_agent,
        parallel_config=ParallelConfig(
            merge_strategy=merge_strategy,
            fail_strategy=fail_strategy,
            timeout=timeout,
        ),
    )
