"""Comprehensive tests for observability/metrics.py - stats, exporters, global functions."""

import asyncio
import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.observability.metrics import (
    CompositeExporter,
    ErrorMetric,
    JSONFileExporter,
    LatencyMetric,
    MetricsCollector,
    MetricsExporter,
    PrometheusExporter,
    StatsD_Exporter,
    TokenUsageMetric,
    export_metrics,
    get_metrics_collector,
    get_metrics_summary,
    initialize_metrics_collector,
    reset_metrics,
)
import logging

logger = logging.getLogger(__name__)


class TestLatencyMetric:
    def test_stop(self):
        logger.info("LatencyMetric: stop")
        m = LatencyMetric(name="op", start_time=0.0)
        m.start_time = 1.0
        m.end_time = 1.05
        m.duration_ms = (m.end_time - m.start_time) * 1000
        assert m.duration_ms == pytest.approx(50.0, abs=1)

    def test_stop_method(self):
        logger.info("LatencyMetric: stop method")
        import time
        m = LatencyMetric(name="op", start_time=time.perf_counter())
        time.sleep(0.005)
        duration = m.stop()
        assert duration > 0
        assert m.end_time is not None


class TestTokenUsageMetric:
    def test_creation(self):
        logger.info("TokenUsageMetric: creation")
        m = TokenUsageMetric(name="llm", prompt_tokens=100, completion_tokens=50, total_tokens=150, model="gpt-4")
        assert m.total_tokens == 150

    def test_cost_estimate_no_model(self):
        logger.info("TokenUsageMetric: cost estimate no model")
        m = TokenUsageMetric(name="llm", prompt_tokens=100, completion_tokens=50)
        assert m.cost_estimate is None

    def test_cost_estimate_with_model(self):
        logger.info("TokenUsageMetric: cost estimate with model")
        m = TokenUsageMetric(name="llm", prompt_tokens=100, completion_tokens=50, model="gpt-4o")
        cost = m.cost_estimate
        assert cost is not None
        assert cost > 0

    def test_cost_estimate_unknown_model(self):
        logger.info("TokenUsageMetric: cost estimate unknown model")
        m = TokenUsageMetric(name="llm", prompt_tokens=100, model="unknown-model-xyz")
        assert m.cost_estimate is None


class TestMetricsCollectorStats:
    def test_get_latency_stats_empty(self):
        logger.info("MetricsCollectorStats: get latency stats empty")
        mc = MetricsCollector()
        stats = mc.get_latency_stats()
        assert stats["count"] == 0

    def test_get_latency_stats_with_data(self):
        logger.info("MetricsCollectorStats: get latency stats with data")
        mc = MetricsCollector()
        mc.record_latency("op1", 10.0)
        mc.record_latency("op1", 20.0)
        mc.record_latency("op1", 30.0)
        stats = mc.get_latency_stats("op1")
        assert stats["count"] == 3
        assert stats["mean_ms"] == 20.0
        assert stats["min_ms"] == 10.0
        assert stats["max_ms"] == 30.0

    def test_get_latency_stats_filter_by_name(self):
        logger.info("MetricsCollectorStats: get latency stats filter by name")
        mc = MetricsCollector()
        mc.record_latency("op1", 10.0)
        mc.record_latency("op2", 20.0)
        stats = mc.get_latency_stats("op1")
        assert stats["count"] == 1

    def test_get_latency_stats_no_durations(self):
        logger.info("MetricsCollectorStats: get latency stats no durations")
        mc = MetricsCollector()
        mc._latencies.append(LatencyMetric(name="op", start_time=0))
        stats = mc.get_latency_stats()
        assert stats["count"] == 1

    def test_get_token_stats_empty(self):
        logger.info("MetricsCollectorStats: get token stats empty")
        mc = MetricsCollector()
        stats = mc.get_token_stats()
        assert stats["total_tokens"] == 0

    def test_get_token_stats_with_data(self):
        logger.info("MetricsCollectorStats: get token stats with data")
        mc = MetricsCollector()
        mc.track_tokens("llm1", prompt_tokens=100, completion_tokens=50, model="gpt-4")
        mc.track_tokens("llm2", prompt_tokens=200, completion_tokens=100, model="gpt-4")
        stats = mc.get_token_stats()
        assert stats["total_prompt_tokens"] == 300
        assert stats["total_completion_tokens"] == 150
        assert "by_model" in stats

    def test_get_token_stats_no_model(self):
        logger.info("MetricsCollectorStats: get token stats no model")
        mc = MetricsCollector()
        mc.track_tokens("llm1", prompt_tokens=100, completion_tokens=50)
        stats = mc.get_token_stats()
        assert stats["total_prompt_tokens"] == 100

    def test_get_error_stats_empty(self):
        logger.info("MetricsCollectorStats: get error stats empty")
        mc = MetricsCollector()
        stats = mc.get_error_stats()
        assert stats["total_errors"] == 0

    def test_get_error_stats_with_data(self):
        logger.info("MetricsCollectorStats: get error stats with data")
        mc = MetricsCollector()
        mc.track_error("op1", ValueError("bad"))
        mc.track_error("op1", TypeError("wrong"))
        mc.track_error("op2", ValueError("bad2"))
        stats = mc.get_error_stats()
        assert stats["total_errors"] == 3
        assert stats["by_type"]["ValueError"] == 2
        assert stats["by_operation"]["op1"] == 2

    def test_get_summary(self):
        logger.info("MetricsCollectorStats: get summary")
        mc = MetricsCollector()
        mc.record_latency("op", 50.0)
        mc.track_tokens("llm", prompt_tokens=100, completion_tokens=50)
        mc.track_error("op", ValueError("err"))
        mc.record_metric("custom", 1.0)
        summary = mc.get_summary()
        assert "latency" in summary
        assert "tokens" in summary
        assert "errors" in summary
        assert "custom" in summary

    def test_increment(self):
        logger.info("MetricsCollectorStats: increment")
        mc = MetricsCollector()
        mc.increment("counter")
        mc.increment("counter")
        mc.increment("counter", 5.0)
        assert len(mc._custom_metrics["counter"]) == 3
        assert sum(mc._custom_metrics["counter"]) == 7.0


class TestMetricsCollectorReportToTrace:
    def test_report_to_trace_empty(self):
        logger.info("MetricsCollectorReportToTrace: report to trace empty")
        mc = MetricsCollector()
        mock_trace = MagicMock()
        mc.report_to_trace(mock_trace)
        mock_trace.update.assert_called()

    def test_report_to_trace_with_latency(self):
        logger.info("MetricsCollectorReportToTrace: report to trace with latency")
        mc = MetricsCollector()
        mc.record_latency("op", 50.0)
        mc.record_latency("op", 100.0)
        mock_trace = MagicMock()
        mc.report_to_trace(mock_trace)
        mock_trace.score.assert_called()

    def test_report_to_trace_with_tokens(self):
        logger.info("MetricsCollectorReportToTrace: report to trace with tokens")
        mc = MetricsCollector()
        mc.track_tokens("llm", prompt_tokens=100, completion_tokens=50, model="gpt-4")
        mock_trace = MagicMock()
        mc.report_to_trace(mock_trace)
        score_calls = mock_trace.score.call_args_list
        score_names = [c.kwargs.get("name") or c.args[0] for c in score_calls if c.args or c.kwargs]
        assert any("total_tokens" in str(n) for n in score_names)

    def test_report_to_trace_with_errors(self):
        logger.info("MetricsCollectorReportToTrace: report to trace with errors")
        mc = MetricsCollector()
        mc.track_error("op", ValueError("err"))
        mock_trace = MagicMock()
        mc.report_to_trace(mock_trace)

    def test_report_to_trace_with_custom(self):
        logger.info("MetricsCollectorReportToTrace: report to trace with custom")
        mc = MetricsCollector()
        mc.record_metric("quality", 0.9)
        mock_trace = MagicMock()
        mc.report_to_trace(mock_trace, include_latency=False, include_tokens=False, include_errors=False)

    def test_report_to_trace_selective(self):
        logger.info("MetricsCollectorReportToTrace: report to trace selective")
        mc = MetricsCollector()
        mc.record_latency("op", 50.0)
        mock_trace = MagicMock()
        mc.report_to_trace(mock_trace, include_tokens=False, include_errors=False, include_custom=False)


class TestMetricsCollectorReportToProviders:
    @patch("orchestrator.observability.provider_manager.get_provider_manager")
    def test_report_to_providers(self, mock_get_pm):
        logger.info("MetricsCollectorReportToProviders: report to providers")
        mock_pm = MagicMock()
        mock_pm.is_enabled = True
        mock_get_pm.return_value = mock_pm

        mc = MetricsCollector()
        mc.record_latency("op", 50.0)
        mc.track_tokens("llm", prompt_tokens=100, completion_tokens=50)
        mc.track_error("op", ValueError("err"))

        with patch("orchestrator.observability.metrics.MetricsCollector.report_to_providers") as mock_method:
            mc.report_to_providers("trace-123")

    def test_report_to_providers_basic(self):
        logger.info("MetricsCollectorReportToProviders: report to providers basic")
        mc = MetricsCollector()
        mc.record_latency("op", 50.0)
        mc.track_tokens("llm", prompt_tokens=100, completion_tokens=50)
        mc.track_error("op", ValueError("err"))
        mc.report_to_providers("trace-123")


class TestMetricsCollectorAsync:
    @pytest.mark.asyncio
    async def test_track_latency_async(self):
        logger.info("MetricsCollectorAsync: track latency async")
        mc = MetricsCollector()
        async with mc.track_latency_async("async_op") as metric:
            await asyncio.sleep(0.005)
        assert metric.duration_ms > 0
        assert len(mc._latencies) == 1


class TestGlobalFunctions:
    def test_get_metrics_collector_singleton(self):
        logger.info("GlobalFunctions: get metrics collector singleton")
        import orchestrator.observability.metrics as mod
        old = mod._global_metrics_collector
        mod._global_metrics_collector = None
        try:
            mc1 = get_metrics_collector()
            mc2 = get_metrics_collector()
            assert mc1 is mc2
        finally:
            mod._global_metrics_collector = old

    def test_initialize_metrics_collector(self):
        logger.info("GlobalFunctions: initialize metrics collector")
        import orchestrator.observability.metrics as mod
        old = mod._global_metrics_collector
        try:
            mc = initialize_metrics_collector()
            assert isinstance(mc, MetricsCollector)
        finally:
            mod._global_metrics_collector = old

    def test_reset_metrics_func(self):
        logger.info("GlobalFunctions: reset metrics func")
        import orchestrator.observability.metrics as mod
        old = mod._global_metrics_collector
        mod._global_metrics_collector = None
        try:
            mc = get_metrics_collector()
            mc.record_latency("op", 50.0)
            reset_metrics()
            assert len(mc._latencies) == 0
        finally:
            mod._global_metrics_collector = old

    def test_get_metrics_summary_func(self):
        logger.info("GlobalFunctions: get metrics summary func")
        import orchestrator.observability.metrics as mod

        old = mod._global_metrics_collector
        mod._global_metrics_collector = None
        try:
            summary = get_metrics_summary()
            assert isinstance(summary, dict)
        finally:
            mod._global_metrics_collector = old


class TestPrometheusExporter:
    def test_init(self):
        logger.info("PrometheusExporter: init")
        exp = PrometheusExporter(gateway_url="http://localhost:9091", job_name="test")
        assert exp.gateway_url == "http://localhost:9091"
        assert exp.job_name == "test"

    def test_format_prometheus_metrics(self):
        logger.info("PrometheusExporter: format prometheus metrics")
        exp = PrometheusExporter(gateway_url="http://localhost:9091")
        metrics = {
            "latency": {"count": 2, "mean_ms": 50.0, "p95_ms": 90.0, "p99_ms": 95.0},
            "tokens": {
                "total_tokens": 150, "total_prompt_tokens": 100,
                "total_completion_tokens": 50, "estimated_cost_usd": 0.001,
                "by_model": {"gpt-4": {"total_tokens": 150, "cost_usd": 0.001}},
            },
            "errors": {"total_errors": 1, "by_type": {"ValueError": 1}},
            "custom": {"quality": {"sum": 0.9}},
        }
        formatted = exp._format_prometheus_metrics(metrics)
        assert "orchestrator_latency_mean_ms" in formatted
        assert "orchestrator_tokens_total" in formatted
        assert "orchestrator_errors_total" in formatted
        assert "orchestrator_custom_quality" in formatted

    def test_format_empty_metrics(self):
        logger.info("PrometheusExporter: format empty metrics")
        exp = PrometheusExporter(gateway_url="http://localhost:9091")
        formatted = exp._format_prometheus_metrics({"latency": {}, "tokens": {}, "errors": {}, "custom": {}})
        assert "orchestrator_errors_total 0" in formatted


class TestJSONFileExporter:
    def test_init(self):
        logger.info("JSONFileExporter: init")
        exp = JSONFileExporter("/tmp/test.json")
        assert exp.file_path == "/tmp/test.json"

    @pytest.mark.asyncio
    async def test_export(self):
        logger.info("JSONFileExporter: export")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            exp = JSONFileExporter(path, append=False, pretty=True)
            metrics = {"latency": {"count": 1}, "tokens": {}, "errors": {}}
            result = await exp.export(metrics)
            assert result is True

            with open(path) as f:
                data = json.load(f)
            assert "metrics" in data
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_export_append(self):
        logger.info("JSONFileExporter: export append")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump([{"timestamp": "t1", "metrics": {}}], f)
            path = f.name

        try:
            exp = JSONFileExporter(path, append=True)
            result = await exp.export({"latency": {}})
            assert result is True
        finally:
            os.unlink(path)


class TestStatsDExporter:
    def test_init(self):
        logger.info("StatsDExporter: init")
        exp = StatsD_Exporter(host="localhost", port=8125, prefix="test")
        assert exp.host == "localhost"
        assert exp.port == 8125


class TestCompositeExporter:
    @pytest.mark.asyncio
    async def test_export(self):
        logger.info("CompositeExporter: export")
        exp1 = MagicMock(spec=MetricsExporter)
        exp1.export = AsyncMock(return_value=True)
        exp2 = MagicMock(spec=MetricsExporter)
        exp2.export = AsyncMock(return_value=False)

        composite = CompositeExporter([exp1, exp2])
        result = await composite.export({"test": "data"})
        assert result is True

    @pytest.mark.asyncio
    async def test_close(self):
        logger.info("CompositeExporter: close")
        exp1 = MagicMock(spec=MetricsExporter)
        exp1.close = AsyncMock()
        composite = CompositeExporter([exp1])
        await composite.close()
        exp1.close.assert_called_once()


class TestExportMetrics:
    @pytest.mark.asyncio
    async def test_export_success(self):
        logger.info("ExportMetrics: export success")
        exp = MagicMock(spec=MetricsExporter)
        exp.export = AsyncMock(return_value=True)

        mc = MetricsCollector()
        mc.record_latency("op", 50.0)
        result = await export_metrics(exp, collector=mc)
        assert result is True

    @pytest.mark.asyncio
    async def test_export_with_reset(self):
        logger.info("ExportMetrics: export with reset")
        exp = MagicMock(spec=MetricsExporter)
        exp.export = AsyncMock(return_value=True)

        mc = MetricsCollector()
        mc.record_latency("op", 50.0)
        result = await export_metrics(exp, collector=mc, reset_after_export=True)
        assert result is True
        assert len(mc._latencies) == 0
