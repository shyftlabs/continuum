"""
Unit tests for the three new coordination workflow agents:
- SupervisedSequentialAgent
- ScatterAgent
- DebateAgent  (including DebateConfig / summarise_arguments)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.agent.base import BaseAgent
from orchestrator.agent.types import AgentResponse, ResponseStatus, TokenUsage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_agent(name: str = "worker", instructions: str = "You are helpful.") -> BaseAgent:
    from orchestrator.agent.config import AgentConfig, AgentMemoryConfig
    return BaseAgent(
        name=name,
        instructions=instructions,
        config=AgentConfig(log_to_session=False),
        memory_config=AgentMemoryConfig(search_memories=False, store_memories=False),
    )


def agent_response(content: str = "result", tokens: int = 10) -> AgentResponse:
    return AgentResponse(
        content=content,
        agent_name="worker",
        status=ResponseStatus.SUCCESS,
        usage=TokenUsage(tokens, tokens, tokens * 2),
    )


def llm_response(content: str = "ok", tokens: int = 5):
    resp = MagicMock()
    resp.content = content
    usage = MagicMock()
    usage.prompt_tokens = tokens
    usage.completion_tokens = tokens
    usage.total_tokens = tokens * 2
    resp.usage = usage
    return resp


def mock_runner(*responses: AgentResponse) -> AsyncMock:
    runner = AsyncMock()
    runner.run.side_effect = list(responses)
    return runner


def mock_context() -> MagicMock:
    ctx = MagicMock()
    ctx.run_id = "test-run-id"
    ctx.metadata = {}
    return ctx


# ===========================================================================
# SupervisedSequentialAgent
# ===========================================================================


class TestSupervisedSequentialAgent:

    def test_import(self):
        from orchestrator.agent.workflow.supervised import SupervisedSequentialAgent
        assert issubclass(SupervisedSequentialAgent, BaseAgent)

    def test_factory_import(self):
        from orchestrator.agent.workflow import create_supervised_agent
        assert callable(create_supervised_agent)

    def test_requires_agents(self):
        from orchestrator.agent.exceptions import AgentConfigurationError
        from orchestrator.agent.workflow.supervised import SupervisedSequentialAgent
        with pytest.raises(AgentConfigurationError):
            SupervisedSequentialAgent(name="s")

    def test_factory_creates_with_config(self):
        from orchestrator.agent.workflow import create_supervised_agent
        from orchestrator.agent.workflow.supervised import SupervisedConfig
        pipeline = create_supervised_agent(
            name="pipe",
            agents=[make_agent("a"), make_agent("b")],
            quality_threshold=0.8,
            max_retries=3,
            supervisor_model="gpt-4o",
        )
        assert pipeline.name == "pipe"
        assert pipeline.supervised_config.quality_threshold == 0.8
        assert pipeline.supervised_config.max_retries == 3
        assert pipeline.supervised_config.supervisor_model == "gpt-4o"

    def test_exported_from_workflow_init(self):
        from orchestrator.agent.workflow import SupervisedSequentialAgent, create_supervised_agent
        assert SupervisedSequentialAgent is not None
        assert callable(create_supervised_agent)

    @pytest.mark.asyncio
    async def test_passes_when_score_above_threshold(self):
        """Step runs once when supervisor scores above threshold — no retries."""
        from orchestrator.agent.workflow.supervised import SupervisedSequentialAgent, SupervisedConfig

        a1 = make_agent("step1")
        pipeline = SupervisedSequentialAgent(
            name="pipe",
            agents=[a1],
            supervised_config=SupervisedConfig(quality_threshold=0.7, max_retries=2),
        )

        runner = mock_runner(agent_response("good output"))
        ctx = mock_context()

        # Supervisor LLM returns score 0.9
        llm = AsyncMock()
        llm.chat.return_value = llm_response("SCORE: 0.9\nFEEDBACK: Excellent.")

        with patch.object(pipeline, "_get_llm", return_value=llm):
            result = await pipeline.execute("do something", runner, ctx)

        assert result.content == "good output"
        assert runner.run.call_count == 1
        assert llm.chat.call_count == 1  # one score call

    @pytest.mark.asyncio
    async def test_retries_when_score_below_threshold(self):
        """Step is retried when score is low; second attempt passes."""
        from orchestrator.agent.workflow.supervised import SupervisedSequentialAgent, SupervisedConfig

        a1 = make_agent("step1")
        pipeline = SupervisedSequentialAgent(
            name="pipe",
            agents=[a1],
            supervised_config=SupervisedConfig(quality_threshold=0.7, max_retries=2),
        )

        runner = mock_runner(
            agent_response("weak output"),
            agent_response("improved output"),
        )
        ctx = mock_context()

        llm = AsyncMock()
        llm.chat.side_effect = [
            llm_response("SCORE: 0.4\nFEEDBACK: Too brief."),   # first attempt — retry
            llm_response("SCORE: 0.85\nFEEDBACK: Much better."), # second attempt — pass
        ]

        with patch.object(pipeline, "_get_llm", return_value=llm):
            result = await pipeline.execute("do something", runner, ctx)

        assert result.content == "improved output"
        assert runner.run.call_count == 2
        assert llm.chat.call_count == 2

    @pytest.mark.asyncio
    async def test_returns_best_after_exhausted_retries(self):
        """When all retries exhausted, returns best output and continues (CONTINUE_ON_ERROR)."""
        from orchestrator.agent.types import FailStrategy
        from orchestrator.agent.workflow.supervised import SupervisedSequentialAgent, SupervisedConfig

        a1 = make_agent("step1")
        pipeline = SupervisedSequentialAgent(
            name="pipe",
            agents=[a1],
            supervised_config=SupervisedConfig(
                quality_threshold=0.9,
                max_retries=1,
                fail_strategy=FailStrategy.CONTINUE_ON_ERROR,
            ),
        )

        runner = mock_runner(
            agent_response("attempt 1"),
            agent_response("attempt 2"),
        )
        ctx = mock_context()

        llm = AsyncMock()
        llm.chat.return_value = llm_response("SCORE: 0.5\nFEEDBACK: Still not great.")

        with patch.object(pipeline, "_get_llm", return_value=llm):
            result = await pipeline.execute("do something", runner, ctx)

        # Returns the best output seen (attempt 1 or 2, both score 0.5)
        assert result.status == ResponseStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_multi_step_pipeline(self):
        """Each step's output is passed as input to the next step."""
        from orchestrator.agent.workflow.supervised import SupervisedSequentialAgent, SupervisedConfig

        a1 = make_agent("step1")
        a2 = make_agent("step2")
        pipeline = SupervisedSequentialAgent(
            name="pipe",
            agents=[a1, a2],
            supervised_config=SupervisedConfig(quality_threshold=0.5, max_retries=0),
        )

        runner = mock_runner(
            agent_response("step1 output"),
            agent_response("step2 output"),
        )
        ctx = mock_context()

        llm = AsyncMock()
        llm.chat.return_value = llm_response("SCORE: 0.8\nFEEDBACK: Good.")

        with patch.object(pipeline, "_get_llm", return_value=llm):
            result = await pipeline.execute("start", runner, ctx)

        assert result.content == "step2 output"
        assert runner.run.call_count == 2
        # Second call should have received step1's output as input
        second_call_input = runner.run.call_args_list[1][1]["input"]
        assert "step1 output" in second_call_input

    @pytest.mark.asyncio
    async def test_no_llm_defaults_to_pass(self):
        """When no LLM is available, scoring defaults to 0.5 and execution continues."""
        from orchestrator.agent.workflow.supervised import SupervisedSequentialAgent, SupervisedConfig

        pipeline = SupervisedSequentialAgent(
            name="pipe",
            agents=[make_agent()],
            supervised_config=SupervisedConfig(quality_threshold=0.4, max_retries=1),
        )

        runner = mock_runner(agent_response("output"))
        ctx = mock_context()

        with patch.object(pipeline, "_get_llm", return_value=None):
            result = await pipeline.execute("task", runner, ctx)

        assert result.content == "output"

    def test_to_dict_includes_supervised_config(self):
        from orchestrator.agent.workflow import create_supervised_agent
        pipeline = create_supervised_agent(
            name="pipe",
            agents=[make_agent()],
            quality_threshold=0.75,
        )
        d = pipeline.to_dict()
        assert d["workflow_type"] == "supervised_sequential"
        assert d["supervised_config"]["quality_threshold"] == 0.75


# ===========================================================================
# ScatterAgent
# ===========================================================================


class TestScatterAgent:

    def test_import(self):
        from orchestrator.agent.workflow.scatter import ScatterAgent
        assert issubclass(ScatterAgent, BaseAgent)

    def test_factory_import(self):
        from orchestrator.agent.workflow import create_scatter_agent
        assert callable(create_scatter_agent)

    def test_requires_agents(self):
        from orchestrator.agent.exceptions import AgentConfigurationError
        from orchestrator.agent.workflow.scatter import ScatterAgent
        with pytest.raises(AgentConfigurationError):
            ScatterAgent(name="s")

    def test_factory_creates_with_config(self):
        from orchestrator.agent.workflow import create_scatter_agent
        scatter = create_scatter_agent(
            name="scatter",
            agents=[make_agent("a"), make_agent("b")],
            split_model="gpt-4o",
            timeout=120,
        )
        assert scatter.name == "scatter"
        assert scatter.scatter_config.split_model == "gpt-4o"
        assert scatter.scatter_config.timeout == 120

    def test_exported_from_workflow_init(self):
        from orchestrator.agent.workflow import ScatterAgent, create_scatter_agent
        assert ScatterAgent is not None
        assert callable(create_scatter_agent)

    @pytest.mark.asyncio
    async def test_uses_explicit_slices(self):
        """When input_slices provided, LLM splitting is skipped."""
        from orchestrator.agent.workflow.scatter import ScatterAgent, ScatterConfig
        from orchestrator.agent.types import MergeStrategy

        a1 = make_agent("a1")
        a2 = make_agent("a2")
        scatter = ScatterAgent(
            name="scatter",
            agents=[a1, a2],
            input_slices=["slice A", "slice B"],
            scatter_config=ScatterConfig(merge_strategy=MergeStrategy.CONCATENATE),
        )

        runner = mock_runner(
            agent_response("result A"),
            agent_response("result B"),
        )
        ctx = mock_context()

        llm = AsyncMock()  # should NOT be called for splitting

        with patch.object(scatter, "_get_llm", return_value=llm):
            result = await scatter.execute("original task", runner, ctx)

        # LLM not called for splitting (only for merge if LLM_SUMMARIZE — here CONCATENATE)
        assert llm.chat.call_count == 0
        # Both agents ran with their respective slices
        assert runner.run.call_count == 2
        call_inputs = [call[1]["input"] for call in runner.run.call_args_list]
        assert "slice A" in call_inputs
        assert "slice B" in call_inputs

    @pytest.mark.asyncio
    async def test_llm_split_called_when_no_explicit_slices(self):
        """LLM splitting is called when input_slices is None."""
        from orchestrator.agent.workflow.scatter import ScatterAgent, ScatterConfig
        from orchestrator.agent.types import MergeStrategy

        a1 = make_agent("financials")
        a2 = make_agent("competitors")
        scatter = ScatterAgent(
            name="scatter",
            agents=[a1, a2],
            scatter_config=ScatterConfig(merge_strategy=MergeStrategy.CONCATENATE),
        )

        runner = mock_runner(
            agent_response("financials result"),
            agent_response("competitors result"),
        )
        ctx = mock_context()

        llm = AsyncMock()
        llm.chat.return_value = llm_response(
            '["Analyse Tesla financials", "Analyse Tesla competitors"]'
        )

        with patch.object(scatter, "_get_llm", return_value=llm):
            result = await scatter.execute("Analyse Tesla", runner, ctx)

        assert llm.chat.call_count == 1  # splitting call
        assert runner.run.call_count == 2

    @pytest.mark.asyncio
    async def test_falls_back_to_same_input_on_bad_llm_json(self):
        """If LLM returns malformed JSON, all agents get the same original input."""
        from orchestrator.agent.workflow.scatter import ScatterAgent, ScatterConfig
        from orchestrator.agent.types import MergeStrategy

        a1 = make_agent("a")
        a2 = make_agent("b")
        scatter = ScatterAgent(
            name="scatter",
            agents=[a1, a2],
            scatter_config=ScatterConfig(merge_strategy=MergeStrategy.CONCATENATE),
        )

        runner = mock_runner(agent_response("r1"), agent_response("r2"))
        ctx = mock_context()

        llm = AsyncMock()
        llm.chat.return_value = llm_response("not json at all")

        with patch.object(scatter, "_get_llm", return_value=llm):
            result = await scatter.execute("original input", runner, ctx)

        # Both agents should have received the original input (fallback)
        call_inputs = [call[1]["input"] for call in runner.run.call_args_list]
        assert all(i == "original input" for i in call_inputs)

    @pytest.mark.asyncio
    async def test_pads_slices_if_fewer_than_agents(self):
        """If fewer explicit slices than agents, pads with original input."""
        from orchestrator.agent.workflow.scatter import ScatterAgent, ScatterConfig
        from orchestrator.agent.types import MergeStrategy

        a1 = make_agent("a")
        a2 = make_agent("b")
        a3 = make_agent("c")
        scatter = ScatterAgent(
            name="scatter",
            agents=[a1, a2, a3],
            input_slices=["slice A", "slice B"],  # only 2, but 3 agents
            scatter_config=ScatterConfig(merge_strategy=MergeStrategy.CONCATENATE),
        )

        runner = mock_runner(
            agent_response("r1"), agent_response("r2"), agent_response("r3")
        )
        ctx = mock_context()

        with patch.object(scatter, "_get_llm", return_value=None):
            await scatter.execute("original", runner, ctx)

        call_inputs = [call[1]["input"] for call in runner.run.call_args_list]
        assert call_inputs[0] == "slice A"
        assert call_inputs[1] == "slice B"
        assert call_inputs[2] == "original"  # padded with original

    def test_to_dict_includes_scatter_config(self):
        from orchestrator.agent.workflow import create_scatter_agent
        scatter = create_scatter_agent(name="s", agents=[make_agent()])
        d = scatter.to_dict()
        assert d["workflow_type"] == "scatter"
        assert "scatter_config" in d


# ===========================================================================
# DebateAgent
# ===========================================================================


class TestDebateAgent:

    def test_import(self):
        from orchestrator.agent.workflow.debate import DebateAgent
        assert issubclass(DebateAgent, BaseAgent)

    def test_factory_import(self):
        from orchestrator.agent.workflow import create_debate_agent
        assert callable(create_debate_agent)

    def test_requires_all_three_agents(self):
        from orchestrator.agent.exceptions import AgentConfigurationError
        from orchestrator.agent.workflow.debate import DebateAgent
        with pytest.raises(AgentConfigurationError):
            DebateAgent(name="d", pro_agent=make_agent(), con_agent=make_agent())

    def test_factory_creates_three_agents(self):
        from orchestrator.agent.workflow import create_debate_agent
        debate = create_debate_agent(
            name="arch",
            pro_stance="Argue FOR microservices.",
            con_stance="Argue FOR monolith.",
        )
        assert debate.pro_agent.name == "arch-pro"
        assert debate.con_agent.name == "arch-con"
        assert debate.judge_agent.name == "arch-judge"

    def test_factory_prepends_topic_description(self):
        from orchestrator.agent.workflow import create_debate_agent
        debate = create_debate_agent(
            name="arch",
            topic_description="We are a 5-engineer team.",
            pro_stance="Argue FOR microservices.",
            con_stance="Argue FOR monolith.",
        )
        assert "5-engineer" in debate.pro_agent.instructions
        assert "5-engineer" in debate.con_agent.instructions
        assert "5-engineer" in debate.judge_agent.instructions

    def test_exported_from_workflow_init(self):
        from orchestrator.agent.workflow import DebateAgent, create_debate_agent
        assert DebateAgent is not None
        assert callable(create_debate_agent)

    @pytest.mark.asyncio
    async def test_basic_debate_flow(self):
        """Pro and con run in parallel; judge receives both arguments."""
        from orchestrator.agent.workflow import create_debate_agent

        debate = create_debate_agent(
            name="test-debate",
            pro_stance="Argue FOR X.",
            con_stance="Argue AGAINST X.",
        )

        runner = AsyncMock()
        runner.run.side_effect = [
            agent_response("pro argument text"),   # pro
            agent_response("con argument text"),   # con
            agent_response("judge synthesis"),     # judge
        ]
        ctx = mock_context()

        result = await debate.execute("Should we do X?", runner, ctx)

        assert result.content == "judge synthesis"
        assert runner.run.call_count == 3
        assert result.agents_used == ["test-debate-pro", "test-debate-con", "test-debate-judge"]

    @pytest.mark.asyncio
    async def test_pro_con_run_before_judge(self):
        """Judge input contains both pro and con arguments."""
        from orchestrator.agent.workflow import create_debate_agent

        debate = create_debate_agent(
            name="d",
            pro_stance="Argue FOR.",
            con_stance="Argue AGAINST.",
        )

        runner = AsyncMock()
        runner.run.side_effect = [
            agent_response("pro says: yes"),
            agent_response("con says: no"),
            agent_response("synthesis"),
        ]
        ctx = mock_context()

        await debate.execute("topic", runner, ctx)

        # Third call = judge; its input should contain both arguments
        judge_input = runner.run.call_args_list[2][1]["input"]
        assert "pro says: yes" in judge_input
        assert "con says: no" in judge_input

    @pytest.mark.asyncio
    async def test_arguments_stored_in_context_metadata(self):
        """Raw pro/con arguments are accessible via context.metadata."""
        from orchestrator.agent.workflow import create_debate_agent

        debate = create_debate_agent(
            name="d",
            pro_stance="FOR",
            con_stance="AGAINST",
        )

        runner = AsyncMock()
        runner.run.side_effect = [
            agent_response("raw pro argument"),
            agent_response("raw con argument"),
            agent_response("synthesis"),
        ]
        ctx = mock_context()

        await debate.execute("topic", runner, ctx)

        assert ctx.metadata["debate_pro"] == "raw pro argument"
        assert ctx.metadata["debate_con"] == "raw con argument"

    @pytest.mark.asyncio
    async def test_continues_when_one_side_fails(self):
        """Judge still runs even if pro agent raises an exception."""
        from orchestrator.agent.workflow import create_debate_agent

        debate = create_debate_agent(
            name="d",
            pro_stance="FOR",
            con_stance="AGAINST",
        )

        call_count = 0

        async def run_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            agent = kwargs["agent"]
            if agent.name == "d-pro":
                raise RuntimeError("pro agent crashed")
            if agent.name == "d-con":
                return agent_response("con argument")
            if agent.name == "d-judge":
                return agent_response("synthesis despite failure")

        runner = AsyncMock()
        runner.run.side_effect = run_side_effect
        ctx = mock_context()

        result = await debate.execute("topic", runner, ctx)

        # Judge still ran and returned a result
        assert result.content == "synthesis despite failure"

    @pytest.mark.asyncio
    async def test_token_usage_accumulates(self):
        """Total tokens = pro + con + judge."""
        from orchestrator.agent.workflow import create_debate_agent

        debate = create_debate_agent(name="d", pro_stance="FOR", con_stance="AGAINST")

        runner = AsyncMock()
        runner.run.side_effect = [
            AgentResponse(content="pro", agent_name="d-pro", status=ResponseStatus.SUCCESS, usage=TokenUsage(10, 10, 20)),
            AgentResponse(content="con", agent_name="d-con", status=ResponseStatus.SUCCESS, usage=TokenUsage(10, 10, 20)),
            AgentResponse(content="synthesis", agent_name="d-judge", status=ResponseStatus.SUCCESS, usage=TokenUsage(20, 20, 40)),
        ]
        ctx = mock_context()

        result = await debate.execute("topic", runner, ctx)
        assert result.usage.total_tokens == 80  # 20 + 20 + 40

    def test_to_dict_includes_all_agents(self):
        from orchestrator.agent.workflow import create_debate_agent
        debate = create_debate_agent(name="d", pro_stance="FOR", con_stance="AGAINST")
        d = debate.to_dict()
        assert d["workflow_type"] == "debate"
        assert d["pro_agent"] == "d-pro"
        assert d["con_agent"] == "d-con"
        assert d["judge_agent"] == "d-judge"
        assert "debate_config" in d


# ===========================================================================
# DebateConfig — summarise_arguments
# ===========================================================================


class TestDebateConfigSummariseArguments:

    def test_defaults(self):
        from orchestrator.agent.workflow.debate import DebateConfig
        cfg = DebateConfig()
        assert cfg.summarise_arguments is False
        assert cfg.truncate_chars == 2000
        assert cfg.summarise_model is None

    def test_opt_in(self):
        from orchestrator.agent.workflow.debate import DebateConfig
        cfg = DebateConfig(summarise_arguments=True, truncate_chars=None)
        assert cfg.summarise_arguments is True
        assert cfg.truncate_chars is None

    def test_factory_accepts_summarise_flag(self):
        from orchestrator.agent.workflow import create_debate_agent
        debate = create_debate_agent(
            name="d",
            pro_stance="FOR",
            con_stance="AGAINST",
            summarise_arguments=True,
            summarise_model="gpt-4o-mini",
            truncate_chars=None,
        )
        assert debate.debate_config.summarise_arguments is True
        assert debate.debate_config.summarise_model == "gpt-4o-mini"
        assert debate.debate_config.truncate_chars is None

    @pytest.mark.asyncio
    async def test_truncation_applied_by_default(self):
        """Without summarise_arguments, long content is truncated at truncate_chars."""
        from orchestrator.agent.workflow.debate import DebateAgent, DebateConfig

        long_pro = "A" * 5000
        long_con = "B" * 5000

        pro_excerpt, con_excerpt, usage = await DebateAgent(
            name="d",
            pro_agent=make_agent("pro"),
            con_agent=make_agent("con"),
            judge_agent=make_agent("judge"),
            debate_config=DebateConfig(summarise_arguments=False, truncate_chars=2000),
        )._prepare_excerpts(long_pro, long_con)

        assert len(pro_excerpt) <= 2001  # 2000 chars + ellipsis
        assert "…" in pro_excerpt
        assert len(con_excerpt) <= 2001

    @pytest.mark.asyncio
    async def test_no_truncation_when_truncate_chars_none(self):
        """truncate_chars=None passes content through unchanged."""
        from orchestrator.agent.workflow.debate import DebateAgent, DebateConfig

        long_pro = "A" * 5000
        long_con = "B" * 5000

        pro_excerpt, con_excerpt, _ = await DebateAgent(
            name="d",
            pro_agent=make_agent("pro"),
            con_agent=make_agent("con"),
            judge_agent=make_agent("judge"),
            debate_config=DebateConfig(summarise_arguments=False, truncate_chars=None),
        )._prepare_excerpts(long_pro, long_con)

        assert pro_excerpt == long_pro
        assert con_excerpt == long_con

    @pytest.mark.asyncio
    async def test_summarise_arguments_calls_llm_for_each_side(self):
        """When summarise_arguments=True, two LLM calls are made (one per side)."""
        from orchestrator.agent.workflow.debate import DebateAgent, DebateConfig

        agent = DebateAgent(
            name="d",
            pro_agent=make_agent("pro"),
            con_agent=make_agent("con"),
            judge_agent=make_agent("judge"),
            debate_config=DebateConfig(summarise_arguments=True),
        )

        llm = AsyncMock()
        llm.chat.side_effect = [
            llm_response("• Pro point 1\n• Pro point 2\n• Pro point 3"),
            llm_response("• Con point 1\n• Con point 2\n• Con point 3"),
        ]

        with patch.object(agent, "_get_llm", return_value=llm):
            pro_excerpt, con_excerpt, usage = await agent._prepare_excerpts(
                "long pro argument " * 200,
                "long con argument " * 200,
            )

        assert llm.chat.call_count == 2
        assert "Pro point 1" in pro_excerpt
        assert "Con point 1" in con_excerpt
        assert usage.total_tokens > 0

    @pytest.mark.asyncio
    async def test_summarise_falls_back_to_truncation_when_no_llm(self):
        """If summarise_arguments=True but no LLM, silently falls back to truncation."""
        from orchestrator.agent.workflow.debate import DebateAgent, DebateConfig

        agent = DebateAgent(
            name="d",
            pro_agent=make_agent("pro"),
            con_agent=make_agent("con"),
            judge_agent=make_agent("judge"),
            debate_config=DebateConfig(summarise_arguments=True, truncate_chars=100),
        )

        with patch.object(agent, "_get_llm", return_value=None):
            pro_excerpt, con_excerpt, _ = await agent._prepare_excerpts(
                "A" * 5000,
                "B" * 5000,
            )

        # Should fall back to truncation, not crash
        assert len(pro_excerpt) <= 101

    @pytest.mark.asyncio
    async def test_summarise_falls_back_on_llm_error(self):
        """If LLM call fails during summarisation, original content is returned."""
        from orchestrator.agent.workflow.debate import DebateAgent, DebateConfig

        agent = DebateAgent(
            name="d",
            pro_agent=make_agent("pro"),
            con_agent=make_agent("con"),
            judge_agent=make_agent("judge"),
            debate_config=DebateConfig(summarise_arguments=True, truncate_chars=None),
        )

        llm = AsyncMock()
        llm.chat.side_effect = RuntimeError("LLM unavailable")

        original_pro = "original pro content"
        original_con = "original con content"

        with patch.object(agent, "_get_llm", return_value=llm):
            pro_excerpt, con_excerpt, _ = await agent._prepare_excerpts(
                original_pro, original_con
            )

        # Falls back to original content on error
        assert pro_excerpt == original_pro
        assert con_excerpt == original_con

    @pytest.mark.asyncio
    async def test_full_debate_with_summarisation(self):
        """End-to-end: summarise_arguments=True compresses before judge sees content."""
        from orchestrator.agent.workflow import create_debate_agent

        debate = create_debate_agent(
            name="d",
            pro_stance="FOR",
            con_stance="AGAINST",
            summarise_arguments=True,
        )

        runner = AsyncMock()
        runner.run.side_effect = [
            agent_response("very long pro argument " * 100),
            agent_response("very long con argument " * 100),
            agent_response("final synthesis"),
        ]

        llm = AsyncMock()
        llm.chat.side_effect = [
            llm_response("• Pro bullet 1\n• Pro bullet 2"),  # summarise pro
            llm_response("• Con bullet 1\n• Con bullet 2"),  # summarise con
        ]

        with patch.object(debate, "_get_llm", return_value=llm):
            result = await debate.execute("Should we do X?", runner, mock_context())

        assert result.content == "final synthesis"
        # Judge input should contain bullet summaries, not the raw long content
        judge_input = runner.run.call_args_list[2][1]["input"]
        assert "Pro bullet 1" in judge_input
        assert "Con bullet 1" in judge_input
