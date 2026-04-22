"""
Parallel Agent - Concurrent execution agent.

Executes multiple agents concurrently and merges their results.

NOTE: Workflow agents now include Langfuse span tracing for full observability.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from orchestrator.agent.base import BaseAgent
from orchestrator.agent.config import ParallelConfig
from orchestrator.agent.exceptions import ParallelWorkflowError
from orchestrator.agent.types import (
    AgentResponse,
    FailStrategy,
    MergeStrategy,
    ResponseStatus,
    RunContext,
    TokenUsage,
)
from orchestrator.logging import get_logger

if TYPE_CHECKING:
    from orchestrator.agent.runner import AgentRunner
    from orchestrator.llm import LLMClient

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
        from orchestrator.agent import BaseAgent
        from orchestrator.agent.workflow import ParallelAgent
        from orchestrator.agent.types import MergeStrategy

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

    def __post_init__(self) -> None:
        """Initialize parallel agent."""
        if not self.name:
            from orchestrator.agent.exceptions import AgentConfigurationError

            raise AgentConfigurationError("Agent name is required")

        if not self.agents:
            from orchestrator.agent.exceptions import AgentConfigurationError

            raise AgentConfigurationError("ParallelAgent requires at least one agent")

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
        # Disable sub-agent Redis saves; one clean pair is saved at the end.
        _orig_log = {a.name: a.config.log_to_session for a in self.agents}
        for a in self.agents:
            a.config.log_to_session = False
        try:
            # Create tasks for all agents
            tasks = []
            for agent in self.agents:
                task = asyncio.create_task(self._run_agent_safe(agent, input_text, runner, context))
                tasks.append((agent.name, task))

            # Wait for all tasks with timeout
            results: dict[str, AgentResponse | Exception] = {}

            try:
                done, pending = await asyncio.wait(
                    [t for _, t in tasks],
                    timeout=self.parallel_config.timeout,
                    return_when=asyncio.ALL_COMPLETED,
                )

                # Cancel pending tasks
                for task in pending:
                    task.cancel()

                # Collect results
                for agent_name, task in tasks:
                    if task.done():
                        try:
                            results[agent_name] = task.result()
                        except Exception as e:
                            results[agent_name] = e
                    else:
                        results[agent_name] = TimeoutError("Task timed out")

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

            if context.session_id:
                await runner.save_turn(
                    session_id=context.session_id,
                    user_message=input_text,
                    assistant_message=merged,
                    agent=None,
                )

            return result
        finally:
            for a in self.agents:
                a.config.log_to_session = _orig_log[a.name]

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
                from orchestrator.core.container import get_container

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

            logger.info("===== FINAL PROMPT [%s/merge] =====\n[user] %s\n========================", self.name, prompt)
            try:
                from orchestrator.llm.config import LLMConfig
                response = await llm_client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    config=LLMConfig(
                        model=self.parallel_config.summary_model or self.model,
                        temperature=0.3,
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
) -> ParallelAgent:
    """
    Factory function to create a parallel agent.

    Args:
        name: Agent name
        agents: List of agents to execute in parallel
        merge_strategy: How to merge results
        fail_strategy: How to handle failures
        timeout: Timeout in seconds

    Returns:
        Configured ParallelAgent
    """
    return ParallelAgent(
        name=name,
        agents=agents,
        parallel_config=ParallelConfig(
            merge_strategy=merge_strategy,
            fail_strategy=fail_strategy,
            timeout=timeout,
        ),
    )
