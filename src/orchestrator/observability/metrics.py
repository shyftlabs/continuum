"""
Metrics collection for observability.

Provides utilities for collecting and reporting metrics via observability providers.

NOTE: Use get_metrics_collector() to get the global collector instance.
The global collector is automatically used by AgentRunner, LLMClient, etc.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from orchestrator.logging import get_logger

if TYPE_CHECKING:
    from orchestrator.observability.tracing import Trace

logger = get_logger(__name__)

# Global metrics collector instance
_global_metrics_collector: MetricsCollector | None = None
_global_lock = threading.Lock()


@dataclass
class LatencyMetric:
    """Latency measurement data."""

    name: str
    start_time: float
    end_time: float | None = None
    duration_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def stop(self) -> float:
        """Stop the timer and return duration in milliseconds."""
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        return self.duration_ms


@dataclass
class TokenUsageMetric:
    """Token usage measurement data."""

    name: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def cost_estimate(self) -> float | None:
        """
        Estimate cost based on model using hardcoded pricing table.
        Pricing is per token in USD.
        """
        if not self.model:
            return None

        # Pricing per token (input, output) in USD
        PRICING: dict[str, tuple[float, float]] = {
            "gpt-4o": (0.0000025, 0.00001),
            "gpt-4o-mini": (0.00000015, 0.0000006),
            "gpt-4o-turbo": (0.0000025, 0.00001),
            "gpt-3.5-turbo": (0.0000005, 0.0000015),
            "gemini-2.5-pro": (0.00000125, 0.000005),
            "gemini-2.5-flash": (0.0000001, 0.0000004),
            "gemini-2.5-flash-lite": (0.0000001, 0.0000004),
            "claude-haiku-4.5": (0.000001, 0.000005),
            "claude-sonnet-4.5": (0.000003, 0.000015),
            "claude-opus-4.5": (0.000015, 0.000075),
        }

        try:
            model_lower = self.model.lower().split("/")[-1]
            pricing = next(
                (v for k, v in PRICING.items() if k in model_lower), None
            )
            if not pricing:
                return None
            input_cost_per_token, output_cost_per_token = pricing
            total = (self.prompt_tokens * input_cost_per_token) + (self.completion_tokens * output_cost_per_token)
            return total if total > 0 else None
        except Exception as e:
            logger.warning(f"Could not calculate cost for model {self.model}: {e}")
            return None


@dataclass
class ErrorMetric:
    """Error tracking data."""

    name: str
    error_type: str
    error_message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and reports metrics to Langfuse.

    Provides utilities for tracking latency, token usage, errors,
    and custom metrics. All metrics are sent to Langfuse as scores
    on the associated trace.

    Example:
        ```python
        from orchestrator.observability import MetricsCollector, TracingManager

        manager = TracingManager()
        metrics = MetricsCollector()

        with manager.trace("my-workflow") as trace:
            # Track latency
            with metrics.track_latency("llm-call"):
                response = await llm.chat(messages)

            # Track tokens
            metrics.track_tokens(
                "llm-call",
                prompt_tokens=100,
                completion_tokens=50,
                model="gpt-4o",
            )

            # Report to Langfuse
            metrics.report_to_trace(trace)
        ```
    """

    def __init__(self):
        """
        Initialize the metrics collector.
        """
        self._client = None
        self._latencies: list[LatencyMetric] = []
        self._token_usage: list[TokenUsageMetric] = []
        self._errors: list[ErrorMetric] = []
        self._custom_metrics: dict[str, list[float]] = {}

    def reset(self) -> None:
        """Clear all collected metrics."""
        self._latencies.clear()
        self._token_usage.clear()
        self._errors.clear()
        self._custom_metrics.clear()

    @contextmanager
    def track_latency(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> Generator[LatencyMetric]:
        """
        Context manager to track latency of an operation.

        Args:
            name: Name of the operation
            metadata: Additional metadata

        Yields:
            LatencyMetric that will have duration_ms set on exit.

        Example:
            ```python
            with metrics.track_latency("database-query") as latency:
                result = db.query(sql)
            print(f"Query took {latency.duration_ms:.2f}ms")
            ```
        """
        metric = LatencyMetric(
            name=name,
            start_time=time.perf_counter(),
            metadata=metadata or {},
        )

        try:
            yield metric
        finally:
            metric.stop()
            self._latencies.append(metric)

    def record_latency(
        self,
        name: str,
        duration_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Record a latency measurement directly.

        Args:
            name: Name of the operation
            duration_ms: Duration in milliseconds
            metadata: Additional metadata
        """
        metric = LatencyMetric(
            name=name,
            start_time=0,
            end_time=0,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )
        self._latencies.append(metric)

    def track_tokens(
        self,
        name: str,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int | None = None,
        model: str | None = None,
    ) -> TokenUsageMetric:
        """
        Track token usage for an LLM call.

        Args:
            name: Name of the operation
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            total_tokens: Total tokens (calculated if not provided)
            model: Model name for cost estimation

        Returns:
            TokenUsageMetric with the recorded data.
        """
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens

        metric = TokenUsageMetric(
            name=name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            model=model,
        )
        self._token_usage.append(metric)
        return metric

    def track_error(
        self,
        name: str,
        error: Exception,
        metadata: dict[str, Any] | None = None,
    ) -> ErrorMetric:
        """
        Track an error occurrence.

        Args:
            name: Name of the operation that failed
            error: The exception that occurred
            metadata: Additional metadata

        Returns:
            ErrorMetric with the recorded data.
        """
        metric = ErrorMetric(
            name=name,
            error_type=type(error).__name__,
            error_message=str(error),
            metadata=metadata or {},
        )
        self._errors.append(metric)
        return metric

    def record_metric(self, name: str, value: float) -> None:
        """
        Record a custom numeric metric.

        Args:
            name: Metric name
            value: Metric value
        """
        if name not in self._custom_metrics:
            self._custom_metrics[name] = []
        self._custom_metrics[name].append(value)

    def increment(self, name: str, value: float = 1.0) -> None:
        """
        Increment a counter metric.

        Args:
            name: Counter name
            value: Amount to increment (default: 1)
        """
        self.record_metric(name, value)

    def get_latency_stats(self, name: str | None = None) -> dict[str, Any]:
        """
        Get latency statistics.

        Args:
            name: Filter by operation name (all if None)

        Returns:
            Dictionary with count, mean, min, max, p50, p95, p99 latencies.
        """
        latencies = self._latencies
        if name:
            latencies = [lat for lat in latencies if lat.name == name]

        if not latencies:
            return {"count": 0}

        durations = [lat.duration_ms for lat in latencies if lat.duration_ms is not None]
        if not durations:
            return {"count": len(latencies)}

        sorted_durations = sorted(durations)
        n = len(sorted_durations)

        return {
            "count": n,
            "mean_ms": sum(durations) / n,
            "min_ms": sorted_durations[0],
            "max_ms": sorted_durations[-1],
            "p50_ms": sorted_durations[n // 2],
            "p95_ms": sorted_durations[int(n * 0.95)] if n > 1 else sorted_durations[-1],
            "p99_ms": sorted_durations[int(n * 0.99)] if n > 1 else sorted_durations[-1],
        }

    def get_token_stats(self) -> dict[str, Any]:
        """
        Get token usage statistics.

        Returns:
            Dictionary with total tokens, costs, and per-model breakdown.
        """
        if not self._token_usage:
            return {"total_prompt_tokens": 0, "total_completion_tokens": 0, "total_tokens": 0}

        total_prompt = sum(t.prompt_tokens for t in self._token_usage)
        total_completion = sum(t.completion_tokens for t in self._token_usage)
        total = sum(t.total_tokens for t in self._token_usage)

        # Calculate costs
        total_cost = 0.0
        for t in self._token_usage:
            cost = t.cost_estimate
            if cost:
                total_cost += cost

        # Per-model breakdown with costs
        by_model: dict[str, dict[str, Any]] = {}
        for t in self._token_usage:
            # Ensure model is properly identified - never use "unknown"
            model = t.model
            if not model:
                # Log warning and skip this entry - model should always be provided
                logger.warning(
                    f"Token usage metric missing model name: {t.name}. "
                    "Model must be provided for proper tracking."
                )
                continue  # Skip entries without model

            if model not in by_model:
                by_model[model] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                    "call_count": 0,
                }

            by_model[model]["prompt_tokens"] += t.prompt_tokens
            by_model[model]["completion_tokens"] += t.completion_tokens
            by_model[model]["total_tokens"] += t.total_tokens
            by_model[model]["call_count"] += 1

            # Add cost for this usage
            cost = t.cost_estimate
            if cost:
                by_model[model]["cost_usd"] += cost

        return {
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total,
            "estimated_cost_usd": round(total_cost, 6),
            "by_model": by_model,
            "model_count": len(by_model),
        }

    def get_error_stats(self) -> dict[str, Any]:
        """
        Get error statistics.

        Returns:
            Dictionary with error counts by type and operation.
        """
        if not self._errors:
            return {"total_errors": 0}

        by_type: dict[str, int] = {}
        by_operation: dict[str, int] = {}

        for e in self._errors:
            by_type[e.error_type] = by_type.get(e.error_type, 0) + 1
            by_operation[e.name] = by_operation.get(e.name, 0) + 1

        return {
            "total_errors": len(self._errors),
            "by_type": by_type,
            "by_operation": by_operation,
        }

    def get_summary(self) -> dict[str, Any]:
        """
        Get a summary of all collected metrics.

        Returns:
            Dictionary with latency, token, error, and custom metric summaries.
        """
        return {
            "latency": self.get_latency_stats(),
            "tokens": self.get_token_stats(),
            "errors": self.get_error_stats(),
            "custom": {
                name: {"count": len(values), "sum": sum(values), "mean": sum(values) / len(values)}
                for name, values in self._custom_metrics.items()
                if values
            },
        }

    def report_to_trace(
        self,
        trace: Trace,
        *,
        include_latency: bool = True,
        include_tokens: bool = True,
        include_errors: bool = True,
        include_custom: bool = True,
    ) -> None:
        """
        Report all collected metrics to an observability trace with comprehensive observability.

        This method provides full auditability by:
        - Adding scores for key metrics
        - Including detailed metadata with per-model breakdown
        - Adding tags for filtering
        - Logging events for audit trail

        Args:
            trace: The trace to report metrics to
            include_latency: Whether to include latency scores
            include_tokens: Whether to include token usage scores
            include_errors: Whether to include error counts
            include_custom: Whether to include custom metrics
        """
        token_stats = self.get_token_stats() if include_tokens else {}
        latency_stats = self.get_latency_stats() if include_latency else {}
        error_stats = self.get_error_stats() if include_errors else {}

        # Build comprehensive metadata
        metadata: dict[str, Any] = {
            "metrics_summary": self.get_summary(),
        }

        # Add per-model breakdown with costs (costs already calculated in get_token_stats)
        if include_tokens and token_stats.get("by_model"):
            # Use the cost_usd already calculated in get_token_stats
            model_breakdown: dict[str, dict[str, Any]] = {}
            for model, usage in token_stats["by_model"].items():
                model_breakdown[model] = {
                    "prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": usage["completion_tokens"],
                    "total_tokens": usage["total_tokens"],
                    "cost_usd": round(usage.get("cost_usd", 0.0), 6),
                    "call_count": usage.get("call_count", 0),
                }

            metadata["model_usage"] = model_breakdown
            metadata["total_cost_usd"] = round(token_stats.get("estimated_cost_usd", 0.0), 6)
            metadata["total_tokens"] = token_stats.get("total_tokens", 0)
            metadata["model_count"] = token_stats.get("model_count", 0)

        # Add latency breakdown
        if include_latency and latency_stats.get("count", 0) > 0:
            metadata["latency"] = {
                "count": latency_stats["count"],
                "mean_ms": round(latency_stats["mean_ms"], 2),
                "p95_ms": round(latency_stats["p95_ms"], 2),
                "p99_ms": round(latency_stats["p99_ms"], 2),
            }

        # Add error breakdown
        if include_errors and error_stats.get("total_errors", 0) > 0:
            metadata["errors"] = {
                "total": error_stats["total_errors"],
                "by_type": error_stats.get("by_type", {}),
                "by_operation": error_stats.get("by_operation", {}),
            }

        # Update trace with comprehensive metadata
        trace.update(metadata=metadata)

        # Add tags for filtering
        tags = ["metrics-reported"]
        if token_stats.get("total_tokens", 0) > 0:
            tags.append("has-llm-usage")
        if error_stats.get("total_errors", 0) > 0:
            tags.append("has-errors")
        if latency_stats.get("count", 0) > 0:
            tags.append("has-latency")

        # Add model tags
        if include_tokens and token_stats.get("by_model"):
            for model in token_stats["by_model"].keys():
                if model and model != "unknown":
                    # Extract provider/model name for tagging
                    provider = model.split("/")[0] if "/" in model else "openai"
                    tags.append(f"provider:{provider}")
                    tags.append(f"model:{model}")

        if tags:
            trace.update(tags=tags)

        # Add scores for key metrics (for dashboard visualization)
        if include_latency and latency_stats.get("count", 0) > 0:
            trace.score(
                name="latency_mean_ms", value=latency_stats["mean_ms"], comment="Mean latency in ms"
            )
            trace.score(
                name="latency_p95_ms", value=latency_stats["p95_ms"], comment="P95 latency in ms"
            )
            trace.score(
                name="latency_p99_ms", value=latency_stats["p99_ms"], comment="P99 latency in ms"
            )

        # Token usage and cost scores
        if include_tokens and token_stats.get("total_tokens", 0) > 0:
            trace.score(
                name="total_tokens",
                value=float(token_stats["total_tokens"]),
                comment="Total tokens used across all models",
            )
            trace.score(
                name="total_prompt_tokens",
                value=float(token_stats["total_prompt_tokens"]),
                comment="Total prompt tokens",
            )
            trace.score(
                name="total_completion_tokens",
                value=float(token_stats["total_completion_tokens"]),
                comment="Total completion tokens",
            )

            if token_stats.get("estimated_cost_usd"):
                trace.score(
                    name="total_cost_usd",
                    value=token_stats["estimated_cost_usd"],
                    comment="Total estimated cost in USD",
                )

            # Per-model scores (use cost already calculated in get_token_stats)
            if token_stats.get("by_model"):
                for model, usage in token_stats["by_model"].items():
                    if model:  # Model should never be None or "unknown" now
                        model_safe = model.replace("/", "_").replace("-", "_")

                        # Token scores per model
                        trace.score(
                            name=f"tokens_{model_safe}",
                            value=float(usage["total_tokens"]),
                            comment=f"Tokens used for {model}",
                        )
                        trace.score(
                            name=f"prompt_tokens_{model_safe}",
                            value=float(usage["prompt_tokens"]),
                            comment=f"Prompt tokens for {model}",
                        )
                        trace.score(
                            name=f"completion_tokens_{model_safe}",
                            value=float(usage["completion_tokens"]),
                            comment=f"Completion tokens for {model}",
                        )

                        # Cost score per model (already calculated)
                        model_cost = usage.get("cost_usd", 0.0)
                        if model_cost > 0:
                            trace.score(
                                name=f"cost_{model_safe}_usd",
                                value=round(model_cost, 6),
                                comment=f"Cost for {model} in USD",
                            )

                        # Call count per model
                        call_count = usage.get("call_count", 0)
                        if call_count > 0:
                            trace.score(
                                name=f"calls_{model_safe}",
                                value=float(call_count),
                                comment=f"Number of calls to {model}",
                            )

        # Error scores
        if include_errors:
            error_count = error_stats.get("total_errors", 0)
            trace.score(
                name="error_count",
                value=float(error_count),
                comment="Total errors encountered",
            )

            # Per-error-type scores
            if error_stats.get("by_type"):
                for error_type, count in error_stats["by_type"].items():
                    trace.score(
                        name=f"error_{error_type.lower()}",
                        value=float(count),
                        comment=f"Count of {error_type} errors",
                    )

        # Custom metrics
        if include_custom:
            for name, values in self._custom_metrics.items():
                if values:
                    trace.score(
                        name=name,
                        value=sum(values),
                        comment=f"Custom metric: {name}",
                    )

        # Log event for audit trail
        trace.event(
            name="metrics_reported",
            input={"summary": self.get_summary()},
            output={"status": "success"},
            metadata={"timestamp": datetime.now(UTC).isoformat()},
        )

    def report_to_providers(
        self,
        trace_id: str,
        observation_id: str | None = None,
    ) -> None:
        """
        Report metrics directly to observability providers.

        Args:
            trace_id: ID of the trace to add scores to
            observation_id: Optional specific observation to score
        """
        try:
            from orchestrator.observability.provider_manager import get_provider_manager

            manager = get_provider_manager()

            if not manager or not manager.is_enabled:
                logger.warning("Cannot report metrics: Provider manager not available")
                return

            stats = self.get_latency_stats()
            if stats.get("count", 0) > 0:
                manager.score(
                    trace_id=trace_id,
                    observation_id=observation_id,
                    name="latency_mean_ms",
                    value=stats["mean_ms"],
                )

            token_stats = self.get_token_stats()
            if token_stats.get("total_tokens", 0) > 0:
                manager.score(
                    trace_id=trace_id,
                    observation_id=observation_id,
                    name="total_tokens",
                    value=float(token_stats["total_tokens"]),
                )

            error_stats = self.get_error_stats()
            manager.score(
                trace_id=trace_id,
                observation_id=observation_id,
                name="error_count",
                value=float(error_stats.get("total_errors", 0)),
            )
        except Exception as e:
            logger.warning(f"Failed to report metrics to providers: {e}")

    @asynccontextmanager
    async def track_latency_async(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncGenerator[LatencyMetric]:
        """
        Async context manager to track latency of an operation.

        Args:
            name: Name of the operation
            metadata: Additional metadata

        Yields:
            LatencyMetric that will have duration_ms set on exit.
        """
        metric = LatencyMetric(
            name=name,
            start_time=time.perf_counter(),
            metadata=metadata or {},
        )

        try:
            yield metric
        finally:
            metric.stop()
            self._latencies.append(metric)


# =============================================================================
# Global MetricsCollector Accessor Functions
# =============================================================================


def get_metrics_collector() -> MetricsCollector:
    """
    Get the global MetricsCollector instance.

    Creates a new instance if one doesn't exist.
    Thread-safe.

    Returns:
        The global MetricsCollector instance.
    """
    global _global_metrics_collector

    if _global_metrics_collector is None:
        with _global_lock:
            if _global_metrics_collector is None:
                _global_metrics_collector = MetricsCollector()
                logger.debug("Global MetricsCollector initialized")

    return _global_metrics_collector


def initialize_metrics_collector() -> MetricsCollector:
    """
    Initialize the global MetricsCollector.

    Returns:
        The initialized MetricsCollector.
    """
    global _global_metrics_collector

    with _global_lock:
        _global_metrics_collector = MetricsCollector()
        logger.info("Global MetricsCollector initialized")

    return _global_metrics_collector


def reset_metrics() -> None:
    """Reset all metrics in the global collector."""
    collector = get_metrics_collector()
    collector.reset()
    logger.debug("Global metrics reset")


def get_metrics_summary() -> dict[str, Any]:
    """Get a summary of all collected metrics from the global collector."""
    return get_metrics_collector().get_summary()


# =============================================================================
# Metrics Export
# =============================================================================


class MetricsExporter:
    """
    Base class for metrics exporters.

    Exporters are responsible for sending metrics to external systems
    like Prometheus, StatsD, CloudWatch, etc.
    """

    async def export(self, metrics: dict[str, Any]) -> bool:
        """
        Export metrics to the target system.

        Args:
            metrics: Dictionary of metrics from MetricsCollector.get_summary()

        Returns:
            True if export was successful, False otherwise.
        """
        raise NotImplementedError("Subclasses must implement export()")

    async def close(self) -> None:
        """Close any connections and cleanup resources."""
        pass


class PrometheusExporter(MetricsExporter):
    """
    Prometheus push gateway exporter.

    Exports metrics to a Prometheus push gateway for scraping.

    Example:
        ```python
        exporter = PrometheusExporter(
            gateway_url="http://localhost:9091",
            job_name="orchestrator"
        )
        await exporter.export(metrics_collector.get_summary())
        ```
    """

    def __init__(
        self,
        gateway_url: str,
        job_name: str = "orchestrator",
        instance: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ):
        """
        Initialize Prometheus exporter.

        Args:
            gateway_url: URL of the Prometheus push gateway
            job_name: Job name for grouping metrics
            instance: Optional instance label
            username: Optional basic auth username
            password: Optional basic auth password
        """
        self.gateway_url = gateway_url.rstrip("/")
        self.job_name = job_name
        self.instance = instance
        self.username = username
        self.password = password

    def _format_prometheus_metrics(self, metrics: dict[str, Any]) -> str:
        """Format metrics in Prometheus exposition format."""
        lines = []

        # Latency metrics
        latency = metrics.get("latency", {})
        if latency.get("count", 0) > 0:
            lines.append("# HELP orchestrator_latency_ms Request latency in milliseconds")
            lines.append("# TYPE orchestrator_latency_ms gauge")
            lines.append(f"orchestrator_latency_mean_ms {latency.get('mean_ms', 0):.2f}")
            lines.append(f"orchestrator_latency_p95_ms {latency.get('p95_ms', 0):.2f}")
            lines.append(f"orchestrator_latency_p99_ms {latency.get('p99_ms', 0):.2f}")
            lines.append(f"orchestrator_latency_count {latency.get('count', 0)}")

        # Token metrics
        tokens = metrics.get("tokens", {})
        if tokens.get("total_tokens", 0) > 0:
            lines.append("# HELP orchestrator_tokens_total Total tokens used")
            lines.append("# TYPE orchestrator_tokens_total counter")
            lines.append(f"orchestrator_tokens_prompt_total {tokens.get('total_prompt_tokens', 0)}")
            lines.append(
                f"orchestrator_tokens_completion_total {tokens.get('total_completion_tokens', 0)}"
            )
            lines.append(f"orchestrator_tokens_total {tokens.get('total_tokens', 0)}")

            # Cost metrics
            if tokens.get("estimated_cost_usd"):
                lines.append("# HELP orchestrator_cost_usd Estimated cost in USD")
                lines.append("# TYPE orchestrator_cost_usd counter")
                lines.append(f"orchestrator_cost_usd {tokens.get('estimated_cost_usd', 0):.6f}")

            # Per-model metrics
            by_model = tokens.get("by_model", {})
            for model, usage in by_model.items():
                lines.append(
                    f'orchestrator_tokens_by_model{{model="{model}"}} {usage.get("total_tokens", 0)}'
                )
                if usage.get("cost_usd"):
                    lines.append(
                        f'orchestrator_cost_by_model_usd{{model="{model}"}} {usage.get("cost_usd", 0):.6f}'
                    )

        # Error metrics
        errors = metrics.get("errors", {})
        lines.append("# HELP orchestrator_errors_total Total errors encountered")
        lines.append("# TYPE orchestrator_errors_total counter")
        lines.append(f"orchestrator_errors_total {errors.get('total_errors', 0)}")

        # Per-error-type metrics
        by_type = errors.get("by_type", {})
        for error_type, count in by_type.items():
            lines.append(f'orchestrator_errors_by_type{{type="{error_type}"}} {count}')

        # Custom metrics
        custom = metrics.get("custom", {})
        for name, data in custom.items():
            metric_name = f"orchestrator_custom_{name.replace('-', '_').replace('.', '_')}"
            lines.append(f"# HELP {metric_name} Custom metric: {name}")
            lines.append(f"# TYPE {metric_name} gauge")
            lines.append(f"{metric_name} {data.get('sum', 0)}")

        return "\n".join(lines)

    async def export(self, metrics: dict[str, Any]) -> bool:
        """Export metrics to Prometheus push gateway."""
        try:
            import aiohttp

            formatted = self._format_prometheus_metrics(metrics)

            url = f"{self.gateway_url}/metrics/job/{self.job_name}"
            if self.instance:
                url += f"/instance/{self.instance}"

            auth = None
            if self.username and self.password:
                auth = aiohttp.BasicAuth(self.username, self.password)

            connector = aiohttp.TCPConnector(enable_cleanup_closed=False)

            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(
                    url,
                    data=formatted,
                    headers={"Content-Type": "text/plain"},
                    auth=auth,
                ) as response:
                    if response.status in (200, 202):
                        logger.debug(f"Exported metrics to Prometheus: {url}")
                        return True
                    else:
                        logger.warning(f"Failed to export metrics to Prometheus: {response.status}")
                        return False

        except ImportError:
            logger.warning("aiohttp not installed, cannot export to Prometheus")
            return False
        except Exception as e:
            logger.error(f"Error exporting metrics to Prometheus: {e}")
            return False


class JSONFileExporter(MetricsExporter):
    """
    Export metrics to a JSON file.

    Useful for debugging, local development, or integration with
    file-based monitoring systems.

    Example:
        ```python
        exporter = JSONFileExporter("/var/log/orchestrator/metrics.json")
        await exporter.export(metrics_collector.get_summary())
        ```
    """

    def __init__(
        self,
        file_path: str,
        append: bool = True,
        pretty: bool = True,
    ):
        """
        Initialize JSON file exporter.

        Args:
            file_path: Path to the JSON file
            append: If True, append to file with timestamp; if False, overwrite
            pretty: If True, format JSON with indentation
        """
        self.file_path = file_path
        self.append = append
        self.pretty = pretty

    async def export(self, metrics: dict[str, Any]) -> bool:
        """Export metrics to JSON file."""
        import json
        from pathlib import Path

        try:
            # Add timestamp
            export_data = {
                "timestamp": datetime.now(UTC).isoformat(),
                "metrics": metrics,
            }

            path = Path(self.file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            if self.append and path.exists():
                # Read existing data
                with open(path) as f:
                    try:
                        existing = json.load(f)
                        if isinstance(existing, list):
                            existing.append(export_data)
                        else:
                            existing = [existing, export_data]
                    except json.JSONDecodeError:
                        existing = [export_data]

                data_to_write = existing
            else:
                data_to_write = [export_data] if self.append else export_data

            with open(path, "w") as f:
                if self.pretty:
                    json.dump(data_to_write, f, indent=2, default=str)
                else:
                    json.dump(data_to_write, f, default=str)

            logger.debug(f"Exported metrics to JSON file: {self.file_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting metrics to JSON file: {e}")
            return False


class StatsD_Exporter(MetricsExporter):
    """
    StatsD metrics exporter.

    Sends metrics to a StatsD server using UDP.

    Example:
        ```python
        exporter = StatsD_Exporter(host="localhost", port=8125)
        await exporter.export(metrics_collector.get_summary())
        ```
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8125,
        prefix: str = "orchestrator",
    ):
        """
        Initialize StatsD exporter.

        Args:
            host: StatsD server host
            port: StatsD server port
            prefix: Metric name prefix
        """
        self.host = host
        self.port = port
        self.prefix = prefix

    async def export(self, metrics: dict[str, Any]) -> bool:
        """Export metrics to StatsD server."""
        import socket

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            messages = []

            # Latency metrics
            latency = metrics.get("latency", {})
            if latency.get("count", 0) > 0:
                messages.append(f"{self.prefix}.latency.mean:{latency.get('mean_ms', 0):.2f}|g")
                messages.append(f"{self.prefix}.latency.p95:{latency.get('p95_ms', 0):.2f}|g")
                messages.append(f"{self.prefix}.latency.p99:{latency.get('p99_ms', 0):.2f}|g")
                messages.append(f"{self.prefix}.latency.count:{latency.get('count', 0)}|c")

            # Token metrics
            tokens = metrics.get("tokens", {})
            if tokens.get("total_tokens", 0) > 0:
                messages.append(
                    f"{self.prefix}.tokens.prompt:{tokens.get('total_prompt_tokens', 0)}|c"
                )
                messages.append(
                    f"{self.prefix}.tokens.completion:{tokens.get('total_completion_tokens', 0)}|c"
                )
                messages.append(f"{self.prefix}.tokens.total:{tokens.get('total_tokens', 0)}|c")
                if tokens.get("estimated_cost_usd"):
                    # StatsD doesn't handle floats well, so multiply by 1M for precision
                    cost_micro = int(tokens.get("estimated_cost_usd", 0) * 1_000_000)
                    messages.append(f"{self.prefix}.cost.usd_micro:{cost_micro}|c")

            # Error metrics
            errors = metrics.get("errors", {})
            messages.append(f"{self.prefix}.errors.total:{errors.get('total_errors', 0)}|c")

            # Send all messages
            data = "\n".join(messages)
            sock.sendto(data.encode(), (self.host, self.port))
            sock.close()

            logger.debug(f"Exported metrics to StatsD: {self.host}:{self.port}")
            return True

        except Exception as e:
            logger.error(f"Error exporting metrics to StatsD: {e}")
            return False


class CompositeExporter(MetricsExporter):
    """
    Composite exporter that sends metrics to multiple destinations.

    Example:
        ```python
        exporter = CompositeExporter([
            PrometheusExporter("http://localhost:9091"),
            JSONFileExporter("/var/log/metrics.json"),
        ])
        await exporter.export(metrics_collector.get_summary())
        ```
    """

    def __init__(self, exporters: list[MetricsExporter]):
        """
        Initialize composite exporter.

        Args:
            exporters: List of exporters to use
        """
        self.exporters = exporters

    async def export(self, metrics: dict[str, Any]) -> bool:
        """Export metrics to all configured exporters."""
        results = await asyncio.gather(
            *[exp.export(metrics) for exp in self.exporters],
            return_exceptions=True,
        )

        success_count = sum(1 for r in results if r is True)
        logger.debug(f"Exported metrics to {success_count}/{len(self.exporters)} exporters")

        return success_count > 0

    async def close(self) -> None:
        """Close all exporters."""
        await asyncio.gather(
            *[exp.close() for exp in self.exporters],
            return_exceptions=True,
        )


async def export_metrics(
    exporter: MetricsExporter,
    collector: MetricsCollector | None = None,
    reset_after_export: bool = False,
) -> bool:
    """
    Export metrics using the specified exporter.

    Convenience function for exporting metrics from the global collector.

    Args:
        exporter: The exporter to use
        collector: Optional specific collector (uses global if None)
        reset_after_export: If True, reset metrics after successful export

    Returns:
        True if export was successful, False otherwise.

    Example:
        ```python
        from orchestrator.observability.metrics import (
            export_metrics,
            PrometheusExporter,
            get_metrics_collector,
        )

        exporter = PrometheusExporter("http://localhost:9091")
        success = await export_metrics(exporter, reset_after_export=True)
        ```
    """
    if collector is None:
        collector = get_metrics_collector()

    metrics = collector.get_summary()
    success = await exporter.export(metrics)

    if success and reset_after_export:
        collector.reset()

    return success
