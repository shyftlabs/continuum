"""
Scatter Agent — Parallel execution with different inputs per branch.

Unlike ParallelAgent (which sends the same input to all agents), ScatterAgent
first uses an LLM to split the task into N slices — one per agent — then runs
each agent on its own slice concurrently. Results are gathered and merged.

This matches the scatter/gather distributed computing pattern, LangGraph's
Send API, and Agno's Coordinate mode.

Usage::

    from continuum.agent.workflow import ScatterAgent, create_scatter_agent

    scatter = create_scatter_agent(
        name="tesla-analysis",
        agents=[financials_agent, competitors_agent, news_agent],
    )

    # LLM splits "Analyse Tesla" into:
    #   financials_agent  → "Analyse Tesla's financial performance and revenue trends"
    #   competitors_agent → "Analyse Tesla's competitors and market positioning"
    #   news_agent        → "Summarise recent Tesla news and strategic announcements"
    result = await runner.run(scatter, "Analyse Tesla")

    # Or provide explicit slices to skip LLM splitting:
    scatter = ScatterAgent(
        name="manual-scatter",
        agents=[agent_a, agent_b],
        input_slices=["Handle financials", "Handle operations"],
    )
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from continuum.agent.base import BaseAgent
from continuum.agent.exceptions import ParallelWorkflowError
from continuum.agent.trace.types import StepKind
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
from continuum.config import settings
from continuum.logging import get_logger
from continuum.observability.trace_context import SpanScope

if TYPE_CHECKING:
    from continuum.agent.runner import AgentRunner
    from continuum.llm import LLMClient

logger = get_logger(__name__)


# =============================================================================
# Config
# =============================================================================


@dataclass
class ScatterConfig:
    """Configuration for ScatterAgent."""

    merge_strategy: MergeStrategy = MergeStrategy.LLM_SUMMARIZE
    fail_strategy: FailStrategy = FailStrategy.CONTINUE_ON_ERROR
    timeout: int = 300
    split_model: str | None = None  # Model for LLM task splitting
    summary_model: str | None = None  # Model for LLM result merging
    summary_prompt: str | None = None  # Custom merge prompt

    def to_dict(self) -> dict[str, Any]:
        return {
            "merge_strategy": self.merge_strategy.value,
            "fail_strategy": self.fail_strategy.value,
            "timeout": self.timeout,
            "split_model": self.split_model,
            "summary_model": self.summary_model,
            "summary_prompt": self.summary_prompt,
        }


# =============================================================================
# Agent
# =============================================================================


@dataclass
class ScatterAgent(BaseAgent):
    """
    Parallel agent that assigns a different input slice to each branch.

    The LLM splits the original task into N focused sub-tasks (one per agent).
    All branches run concurrently. Results are gathered and merged using the
    configured MergeStrategy.

    Alternatively, provide `input_slices` to skip LLM splitting entirely.

    Example::

        scatter = ScatterAgent(
            name="market-analysis",
            agents=[financials_agent, competitors_agent, news_agent],
            # input_slices=["slice A", "slice B", "slice C"]  # optional override
        )
        result = await runner.run(scatter, "Analyse Tesla")
    """

    agents: list[BaseAgent] = field(default_factory=list)
    scatter_config: ScatterConfig = field(default_factory=ScatterConfig)

    # Optional explicit slices — bypasses LLM splitting when provided
    input_slices: list[str] | None = None

    # Agent whose memory_config governs post-execution long-term memory writes.
    # If None (default), no memory is written after scatter-gather completes.
    memory_agent: BaseAgent | None = None

    def __post_init__(self) -> None:
        if not self.name:
            from continuum.agent.exceptions import AgentConfigurationError

            raise AgentConfigurationError("Agent name is required")
        if not self.agents:
            from continuum.agent.exceptions import AgentConfigurationError

            raise AgentConfigurationError("ScatterAgent requires at least one agent")

    @publish_active_policy
    async def execute(
        self,
        input_text: str,
        runner: AgentRunner,
        context: RunContext,
        llm_client: LLMClient | None = None,
    ) -> AgentResponse:
        """
        Split input, run agents in parallel on their slices, gather and merge results.

        Args:
            input_text: Original task description
            runner: Agent runner
            context: Run context
            llm_client: Optional LLM client override

        Returns:
            Merged AgentResponse
        """
        if llm_client is None:
            llm_client = self._get_llm()

        # Own the one decision trace spanning scatter branches + the gather stage.
        created = runner.ensure_recorder(context, self.name, input_text)
        context.suppress_session_log = True
        async with SpanScope(
            f"workflow.scatter.{self.name}",
            input={
                "input_preview": input_text[:500],
                "agent_count": len(self.agents),
                "agents": [a.name for a in self.agents],
                "explicit_slices": self.input_slices is not None,
            },
            metadata={"workflow_type": "scatter"},
        ) as workflow_span:
            # Step 1 — determine input slices
            slices = await self._get_slices(input_text, llm_client)
            logger.info(
                f"ScatterAgent '{self.name}': {len(slices)} slices for {len(self.agents)} agents"
            )
            workflow_span.add_metadata("slices_preview", [s[:100] for s in slices])

            # Step 2 — scatter: run all branches concurrently into ISOLATED
            # recorders so their steps don't interleave in the shared trace.
            successful, failed = await self._scatter(slices, runner, context)

            if failed:
                logger.warning(f"ScatterAgent: failed branches: {list(failed.keys())}")
                if self.scatter_config.fail_strategy == FailStrategy.REQUIRE_ALL:
                    raise ParallelWorkflowError(
                        f"Some branches failed: {list(failed.keys())}",
                        failed_agents=list(failed.keys()),
                        run_id=context.run_id,
                    )
                if self.scatter_config.fail_strategy == FailStrategy.FAIL_FAST and not successful:
                    raise ParallelWorkflowError(
                        "All branches failed",
                        failed_agents=list(failed.keys()),
                        run_id=context.run_id,
                    )

            if not successful:
                return AgentResponse(
                    content="All scatter branches failed",
                    agent_name=self.name,
                    status=ResponseStatus.ERROR,
                    error="; ".join(f"{k}: {v}" for k, v in failed.items()),
                )

            # Step 3 — gather/merge (stage n_branches; recorded explicitly because
            # _merge_results calls the LLM directly, not via runner.run).
            merged = await self._gather(successful, input_text, llm_client, context)
            total_usage = TokenUsage()
            for resp in successful.values():
                total_usage = total_usage.add(resp.usage)

            workflow_span.set_output(
                {
                    "success": True,
                    "branches_succeeded": len(successful),
                    "branches_failed": len(failed),
                    "total_tokens": total_usage.total_tokens,
                }
            )

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
                agent=self.memory_agent,
            )

        result.run_id = context.run_id
        if created:
            await runner.persist_decision_trace(context, result)
        return result

    async def _scatter(
        self,
        slices: list[str],
        runner: AgentRunner,
        context: RunContext,
    ) -> tuple[dict[str, AgentResponse], dict[str, str]]:
        """Run every branch concurrently into its own isolated recorder, then
        absorb each branch's steps back into the shared trace IN BRANCH ORDER.

        Returns ``(successful, failed)`` keyed by agent name. The absorb pass runs
        in deterministic ``self.agents`` order so the merged trace is segmentable
        (one ``WORKFLOW_STEP`` marker per branch, contiguous stage-indexed steps)
        regardless of wall-clock interleaving.
        """
        branch_recs: list[Any] = []
        tasks = []
        for i, (agent, slice_input) in enumerate(zip(self.agents, slices, strict=False)):
            branch_ctx, branch_rec = branch_recorder_context(context, index=i)
            branch_recs.append(branch_rec)
            tasks.append(
                asyncio.create_task(self._run_agent_safe(agent, slice_input, runner, branch_ctx))
            )

        try:
            _done, pending = await asyncio.wait(
                tasks,
                timeout=self.scatter_config.timeout,
                return_when=asyncio.ALL_COMPLETED,
            )
            for task in pending:
                task.cancel()
        except Exception as e:
            raise ParallelWorkflowError(
                f"Scatter execution failed: {e}",
                run_id=context.run_id,
                original_error=e,
            ) from e

        successful: dict[str, AgentResponse] = {}
        failed: dict[str, str] = {}
        for i, (agent, task) in enumerate(zip(self.agents, tasks, strict=False)):
            if task.done():
                try:
                    successful[agent.name] = task.result()
                except Exception as e:
                    failed[agent.name] = str(e)
            else:
                failed[agent.name] = "Timed out"

            # Ordered capture: merge this branch's isolated trace at stage i.
            branch_rec = branch_recs[i]
            if context.recorder is not None and branch_rec is not None:
                context.recorder.absorb(
                    branch_rec.trace.steps,
                    stage=i,
                    label=agent.name,
                    orchestrator_name=self.name,
                )

        return successful, failed

    async def _gather(
        self,
        successful: dict[str, AgentResponse],
        original_input: str,
        llm_client: Any | None,
        context: RunContext,
    ) -> str:
        """Merge branch results (the gather stage) and record it at stage
        ``len(self.agents)``. The merge LLM call doesn't go through ``runner.run``,
        so the marker + output are recorded directly on the shared recorder."""
        merged = await self._merge_results(successful, original_input, llm_client)
        if context.recorder is not None:
            gather_stage = len(self.agents)
            context.recorder.record_workflow_step(
                self.name, stage=gather_stage, label="gather", agent_stack=[self.name]
            )
            context.recorder.record(
                StepKind.LLM_CALL,
                self.name,
                turn=0,
                agent_stack=[self.name],
                decision={"merge_strategy": self.scatter_config.merge_strategy.value},
                output=merged,
            )
        return merged

    async def resume_from(
        self,
        *,
        parent_trace: Any,
        from_step: str,
        override: dict[str, Any] | None,
        runner: AgentRunner,
        context: RunContext,
    ) -> AgentResponse:
        """Re-run one scattered branch (or the gather stage) and re-gather.

        Forking branch ``k`` re-runs only that branch with ``override`` applied;
        the sibling branches' outputs are recovered from ``parent_trace`` and the
        SAME gather/reduce produces the final result. Forking the gather stage
        (stage ``n``) re-runs the merge over all cached branch outputs.
        """
        step_stage, stage_first = segment_by_markers(parent_trace)
        stage_idx = step_stage.get(from_step)
        if stage_idx is None:
            raise ValueError(f"resume_from: step '{from_step}' not found in trace")

        n = len(self.agents)
        cached = branch_outputs_from_trace(parent_trace)

        created = runner.ensure_recorder(context, self.name, parent_trace.user_query)
        if created:
            link_lineage(context, parent_trace, from_step, override, stage_idx)
        context.suppress_session_log = True

        llm_client = self._get_llm()

        # Recover every sibling branch's cached output as an AgentResponse.
        successful: dict[str, AgentResponse] = {}
        for i, agent in enumerate(self.agents):
            if i in cached:
                successful[agent.name] = AgentResponse(
                    content=cached[i],
                    agent_name=agent.name,
                    status=ResponseStatus.SUCCESS,
                )

        if stage_idx < n:
            # Re-run only the forked branch into an isolated recorder, absorb at
            # its stage index, and replace its cached entry.
            agent = self.agents[stage_idx]
            new_input = resumed_input(stage_first.get(stage_idx), override, parent_trace.user_query)
            branch_ctx, branch_rec = branch_recorder_context(context, index=stage_idx)
            resp = await self._run_agent_safe(agent, new_input, runner, branch_ctx)
            successful[agent.name] = resp
            if context.recorder is not None and branch_rec is not None:
                context.recorder.absorb(
                    branch_rec.trace.steps,
                    stage=stage_idx,
                    label=agent.name,
                    orchestrator_name=self.name,
                )

        # Re-gather over the (possibly re-run) branch outputs.
        merged = await self._gather(successful, parent_trace.user_query, llm_client, context)
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
        if created:
            await runner.persist_decision_trace(context, result)
        return result

    async def _get_slices(
        self,
        input_text: str,
        llm_client: Any | None,
    ) -> list[str]:
        """
        Return one input slice per agent.

        Uses `input_slices` if provided, otherwise asks the LLM to split the task.
        Falls back to giving all agents the same input if splitting fails.
        """
        if self.input_slices is not None:
            n = len(self.agents)
            slices = list(self.input_slices)
            while len(slices) < n:
                slices.append(input_text)
            return slices[:n]

        if not llm_client:
            logger.warning("ScatterAgent: no LLM client — using same input for all branches")
            return [input_text] * len(self.agents)

        return await self._llm_split(input_text, llm_client)

    async def _llm_split(self, input_text: str, llm_client: Any) -> list[str]:
        """Ask the LLM to decompose the task into N focused sub-tasks."""
        from continuum.llm.config import LLMConfig

        n = len(self.agents)
        agent_names = [a.name for a in self.agents]
        model = self.scatter_config.split_model or settings.default_llm_model

        prompt = (
            f"Split this task into exactly {n} short sub-tasks, one per agent.\n\n"
            f"Task: {input_text}\n\n"
            f"Agents: {', '.join(agent_names)}\n\n"
            f"Rules:\n"
            f"- Each sub-task is a single sentence\n"
            f"- No overlap — each covers a distinct item or aspect\n"
            f"- Return ONLY a JSON array of {n} strings\n\n"
            f'Example: ["Sub-task 1", "Sub-task 2", "Sub-task 3"]'
        )

        try:
            response = await llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                config=LLMConfig(model=model, temperature=0.2, max_tokens=1200),
                auto_session=False,
            )

            content = (response.content or "").strip()
            logger.info(f"ScatterAgent: raw split response: {content[:500]}")
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            slices: list[str] = json.loads(content)

            if not isinstance(slices, list) or len(slices) != len(self.agents):
                raise ValueError(f"Expected {len(self.agents)} slices, got {len(slices)}")

            logger.info(f"ScatterAgent: LLM split into {len(slices)} slices")
            for i, (agent, s) in enumerate(zip(self.agents, slices, strict=False)):
                logger.debug(f"  branch {i + 1} ({agent.name}): {s[:100]}")

            return slices

        except Exception as e:
            logger.warning(
                f"ScatterAgent: LLM splitting failed ({type(e).__name__}: {e}) — using same input for all branches"
            )
            return [input_text] * len(self.agents)

    async def _run_agent_safe(
        self,
        agent: BaseAgent,
        input_text: str,
        runner: AgentRunner,
        context: RunContext,
    ) -> AgentResponse:
        """Run a single branch with error handling."""
        try:
            return await runner.run(agent=agent, input=input_text, context=context)
        except Exception as e:
            logger.error(f"Scatter branch '{agent.name}' failed: {e}")
            raise

    async def _merge_results(
        self,
        results: dict[str, AgentResponse],
        original_input: str,
        llm_client: Any | None,
    ) -> str:
        """Merge branch results using the configured strategy."""
        strategy = self.scatter_config.merge_strategy

        if strategy == MergeStrategy.FIRST_SUCCESS:
            return next(iter(results.values())).content or ""

        if strategy == MergeStrategy.CONCATENATE:
            return "\n\n".join(f"## {name}\n{resp.content}" for name, resp in results.items())

        if strategy == MergeStrategy.STRUCTURED:
            return json.dumps({name: resp.content for name, resp in results.items()}, indent=2)

        # LLM_SUMMARIZE (default)
        if llm_client is None:
            try:
                from continuum.core.container import get_container

                llm_client = get_container().llm_client
            except Exception:
                pass

        if llm_client is None:
            return "\n\n".join(f"## {name}\n{resp.content}" for name, resp in results.items())

        outputs = "\n\n".join(f"### {name}\n{resp.content}" for name, resp in results.items())
        if self.scatter_config.summary_prompt:
            prompt = f"{outputs}\n\n{self.scatter_config.summary_prompt}"
        else:
            prompt = (
                f"Multiple specialist agents worked on different aspects of this task:\n\n"
                f"Original task: {original_input}\n\n"
                f"Specialist outputs:\n{outputs}\n\n"
                f"Synthesise these into a single coherent response that integrates all perspectives."
            )

        logger.info(
            "===== FINAL PROMPT [%s/merge] =====\n[user] %s\n========================",
            self.name,
            prompt,
        )

        from continuum.llm.config import LLMConfig

        try:
            response = await llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                config=LLMConfig(
                    model=self.scatter_config.summary_model or self.model,
                    temperature=0.3,
                ),
                auto_session=False,
            )
            return response.content or ""
        except Exception as e:
            logger.warning(f"ScatterAgent: LLM merge failed ({e}) — concatenating")
            return "\n\n".join(f"## {name}\n{resp.content}" for name, resp in results.items())

    def _get_llm(self) -> Any | None:
        try:
            from continuum.core.container import get_container

            return get_container().llm_client
        except Exception:
            return None

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "agents": [a.name for a in self.agents],
                "scatter_config": self.scatter_config.to_dict(),
                "input_slices": self.input_slices,
                "workflow_type": "scatter",
            }
        )
        return base


# =============================================================================
# Factory
# =============================================================================


def create_scatter_agent(
    name: str,
    agents: list[BaseAgent],
    *,
    input_slices: list[str] | None = None,
    merge_strategy: MergeStrategy = MergeStrategy.LLM_SUMMARIZE,
    fail_strategy: FailStrategy = FailStrategy.CONTINUE_ON_ERROR,
    split_model: str | None = None,
    timeout: int = 300,
    memory_agent: BaseAgent | None = None,
) -> ScatterAgent:
    """
    Factory for ScatterAgent.

    Args:
        name: Agent name
        agents: Agents to run in parallel (each gets a different input slice)
        input_slices: Explicit slices — skips LLM splitting when provided
        merge_strategy: How to combine branch results
        fail_strategy: How to handle branch failures
        split_model: LLM model to use for task splitting
        timeout: Overall timeout in seconds

    Returns:
        Configured ScatterAgent

    Example::

        scatter = create_scatter_agent(
            name="tesla-analysis",
            agents=[financials_agent, competitors_agent, news_agent],
        )
        result = await runner.run(scatter, "Analyse Tesla")
    """
    return ScatterAgent(
        name=name,
        agents=agents,
        input_slices=input_slices,
        memory_agent=memory_agent,
        scatter_config=ScatterConfig(
            merge_strategy=merge_strategy,
            fail_strategy=fail_strategy,
            split_model=split_model,
            timeout=timeout,
        ),
    )
