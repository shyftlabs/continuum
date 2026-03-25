"""
Langfuse Datasets client for evaluation regression testing.

Uses the LangfuseClient already wired into the project's observability
pipeline — no second connection to Langfuse is opened.

Typical workflow::

    # 1. Create/ensure the dataset exists
    ds = LangfuseDatasetClient("customer-support-evals")
    ds.ensure_dataset_exists(description="Weekly regression suite")

    # 2. Upload test cases (one-time setup, or programmatically)
    case = EvalCase(input_text="How do I cancel?", expected_output="...")
    item_id = ds.upload_case(case)

    # 3. Run agent + evaluator, then push scores to Langfuse
    response = await runner.run(my_agent, case.input_text)
    result = await evaluator.evaluate(case, response.content)
    ds.upload_scores(result, trace_id=response.trace_id)

    # 4. Or: pull cases from the Langfuse UI-curated dataset
    for case in ds.fetch_cases():
        response = await runner.run(my_agent, case.input_text)
        result = await evaluator.evaluate(case, response.content)
        ds.upload_scores(result, trace_id=response.trace_id)

Scores appear in the Langfuse dashboard under the trace, allowing before/after
comparison across prompt or agent changes.
"""

from __future__ import annotations

from typing import Any

from orchestrator.evaluation.types import EvalCase, EvalResult
from orchestrator.logging import get_logger

logger = get_logger(__name__)


class LangfuseDatasetClient:
    """
    High-level client for managing Langfuse Datasets and uploading eval scores.

    All methods are synchronous — the Langfuse v2 Python SDK is sync.
    Every method is safe to call when Langfuse is disabled or not configured;
    it will log at debug level and return a safe default (None / empty list).

    Args:
        dataset_name:     Langfuse dataset name (created if it does not exist).
        langfuse_client:  Inject a LangfuseClient instance directly. If None,
                          resolved lazily from get_container().langfuse_client.
    """

    def __init__(
        self,
        dataset_name: str,
        *,
        langfuse_client: Any | None = None,
    ) -> None:
        self._dataset_name = dataset_name
        self._lf: Any | None = langfuse_client

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _client(self) -> Any | None:
        """Lazily resolve the LangfuseClient from the container."""
        if self._lf is not None:
            return self._lf
        try:
            from orchestrator.core.container import get_container
            self._lf = get_container().langfuse_client
        except Exception as exc:
            logger.debug(f"LangfuseDatasetClient: could not get container: {exc}")
        return self._lf

    # ------------------------------------------------------------------
    # Dataset management
    # ------------------------------------------------------------------

    def ensure_dataset_exists(
        self,
        *,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """
        Create the dataset in Langfuse if it does not already exist.

        Returns the Langfuse dataset object, or None if Langfuse is disabled.
        """
        client = self._client()
        if client is None:
            logger.debug("LangfuseDatasetClient: Langfuse not configured, skipping")
            return None

        existing = client.get_dataset(self._dataset_name)
        if existing is not None:
            logger.debug(f"LangfuseDatasetClient: dataset '{self._dataset_name}' already exists")
            return existing

        dataset = client.create_dataset(
            name=self._dataset_name,
            description=description,
            metadata=metadata,
        )
        logger.info(f"LangfuseDatasetClient: created dataset '{self._dataset_name}'")
        return dataset

    # ------------------------------------------------------------------
    # Uploading cases
    # ------------------------------------------------------------------

    def upload_case(
        self,
        case: EvalCase,
        *,
        extra_metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """
        Upload an EvalCase as a Langfuse dataset item.

        Args:
            case:           EvalCase to upload.
            extra_metadata: Additional metadata merged with case.metadata.

        Returns:
            The Langfuse dataset item ID, or None if Langfuse is disabled.
            Store this ID in case.metadata["langfuse_item_id"] if you need it
            later for link_run().
        """
        client = self._client()
        if client is None:
            return None

        item_metadata = {**case.metadata, **(extra_metadata or {}), "case_id": case.case_id}
        item = client.create_dataset_item(
            dataset_name=self._dataset_name,
            input={"input_text": case.input_text, "context": case.context},
            expected_output=case.expected_output,
            metadata=item_metadata,
        )

        if item is None:
            return None

        item_id = str(getattr(item, "id", "") or "")
        logger.debug(f"LangfuseDatasetClient: uploaded case '{case.case_id}' → item '{item_id}'")
        return item_id or None

    def upload_bulk_cases(
        self,
        cases: list[EvalCase],
        *,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """
        Upload multiple EvalCases.

        Returns a mapping of case_id → langfuse_item_id for successfully
        uploaded cases.
        """
        mapping: dict[str, str] = {}
        for case in cases:
            item_id = self.upload_case(case, extra_metadata=extra_metadata)
            if item_id:
                mapping[case.case_id] = item_id
        logger.info(
            f"LangfuseDatasetClient: uploaded {len(mapping)}/{len(cases)} cases "
            f"to dataset '{self._dataset_name}'"
        )
        return mapping

    # ------------------------------------------------------------------
    # Linking runs
    # ------------------------------------------------------------------

    def link_run(
        self,
        dataset_item_id: str,
        *,
        trace_id: str,
        run_name: str,
        run_description: str | None = None,
        run_metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Link a completed agent trace to a Langfuse dataset item.

        In Langfuse v2 this is done via DatasetItem.link(). Each linked run
        appears in the Langfuse dashboard under the dataset item, allowing
        side-by-side comparison across different agent versions.

        Args:
            dataset_item_id:  Item ID returned by upload_case().
            trace_id:         Langfuse trace ID from the agent run.
            run_name:         Logical name for this evaluation run (e.g. "v1.3").
            run_description:  Optional human description.
            run_metadata:     Arbitrary metadata for the run record.
        """
        client = self._client()
        if client is None:
            return

        # Try public API first, then fall back to private attribute
        raw_lf = getattr(client, "client", None) or getattr(client, "_client", None)
        if raw_lf is None:
            logger.debug(
                "LangfuseDatasetClient.link_run: raw Langfuse client not available. "
                "Tried 'client' and '_client' attributes."
            )
            return

        try:
            dataset = raw_lf.get_dataset(self._dataset_name)
            item = next(
                (i for i in dataset.items if str(getattr(i, "id", "")) == dataset_item_id),
                None,
            )
            if item is None:
                logger.warning(
                    f"LangfuseDatasetClient.link_run: item '{dataset_item_id}' not found "
                    f"in dataset '{self._dataset_name}'"
                )
                return

            item.link(
                observed_trace_or_observation=raw_lf.get_trace(trace_id),
                run_name=run_name,
                run_description=run_description,
                run_metadata=run_metadata or {},
            )
            logger.debug(
                f"LangfuseDatasetClient: linked trace '{trace_id}' to item "
                f"'{dataset_item_id}' as run '{run_name}'"
            )
        except Exception as exc:
            logger.warning(f"LangfuseDatasetClient.link_run failed: {exc}")

    # ------------------------------------------------------------------
    # Uploading scores
    # ------------------------------------------------------------------

    def upload_scores(
        self,
        result: EvalResult,
        *,
        trace_id: str,
        observation_id: str | None = None,
    ) -> None:
        """
        Push all criterion scores from an EvalResult to Langfuse.

        Each CriterionScore becomes one Langfuse score:
            name    = "<evaluator_name>/<criterion>"
            value   = criterion.score  (float 0–1)
            comment = criterion.reasoning

        Args:
            result:         EvalResult from any evaluator.
            trace_id:       Langfuse trace ID to attach scores to.
            observation_id: Optional span/generation ID for finer attribution.
        """
        client = self._client()
        if client is None:
            return

        for cs in result.scores:
            score_name = f"{result.evaluator_name}/{cs.criterion}"
            client.score(
                trace_id=trace_id,
                observation_id=observation_id,
                name=score_name,
                value=cs.score,
                comment=cs.reasoning or None,
                data_type="NUMERIC",
            )

        # Also upload the overall score if available
        if result.overall_score is not None:
            client.score(
                trace_id=trace_id,
                observation_id=observation_id,
                name=f"{result.evaluator_name}/overall",
                value=result.overall_score,
                comment=f"passed={result.overall_passed}",
                data_type="NUMERIC",
            )

        logger.debug(
            f"LangfuseDatasetClient: uploaded {len(result.scores)} scores "
            f"(+ overall) to trace '{trace_id}'"
        )

    # ------------------------------------------------------------------
    # Fetching cases
    # ------------------------------------------------------------------

    def fetch_cases(self, limit: int | None = None) -> list[EvalCase]:
        """
        Download dataset items from Langfuse as EvalCase objects.

        Useful for running evaluations against a dataset curated in the
        Langfuse UI without re-uploading cases from code.

        Args:
            limit: Maximum number of items to fetch (None = all).

        Returns:
            List of EvalCase objects populated from dataset item data.
        """
        client = self._client()
        if client is None:
            return []

        try:
            dataset = client.get_dataset(self._dataset_name)
            if dataset is None:
                logger.warning(
                    f"LangfuseDatasetClient: dataset '{self._dataset_name}' not found"
                )
                return []

            items = dataset.items
            if limit is not None:
                items = items[:limit]

            cases: list[EvalCase] = []
            for item in items:
                raw_input = getattr(item, "input", {}) or {}
                if isinstance(raw_input, dict):
                    input_text = str(raw_input.get("input_text", ""))
                    context = list(raw_input.get("context", []))
                else:
                    input_text = str(raw_input)
                    context = []

                expected = getattr(item, "expected_output", None)
                item_id = str(getattr(item, "id", "") or "")
                item_meta = dict(getattr(item, "metadata", {}) or {})
                item_meta["langfuse_item_id"] = item_id

                cases.append(
                    EvalCase(
                        input_text=input_text,
                        expected_output=str(expected) if expected is not None else None,
                        context=context,
                        metadata=item_meta,
                    )
                )

            logger.info(
                f"LangfuseDatasetClient: fetched {len(cases)} cases "
                f"from dataset '{self._dataset_name}'"
            )
            return cases

        except Exception as exc:
            logger.warning(f"LangfuseDatasetClient.fetch_cases failed: {exc}")
            return []
