"""Headroom compression integration (sidecar client).

Continuum talks to a locally-run Headroom sidecar (``headroom proxy``) over HTTP
to compress large tool/RAG/log content before it reaches the model. Continuum
never imports the Headroom package or stores originals — this subpackage is only
the thin client + orchestration. See ``gap-analysis/headroom-native-integration-plan.md``.
"""

from __future__ import annotations

from continuum.llm.headroom.client import CompressionStats, HeadroomClient

__all__ = ["CompressionStats", "HeadroomClient"]
