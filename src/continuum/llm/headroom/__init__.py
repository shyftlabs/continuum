"""Headroom compression integration.

Compresses large tool/RAG/log content before it reaches the model. Two backends,
selected by ``settings.headroom_mode`` (both drive the same HeadroomCompressor —
fail-open, per-run anti-forgery, marker-hash capture are backend-agnostic):

  local (default): in-process ``import headroom`` (requires the
      [headroom-local] extra). No sidecar; originals live in the library's
      in-process CCR store. If the extra is missing, fail-open disables
      compression rather than crashing.
  endpoint: HTTP to a locally-run ``headroom proxy`` sidecar. Continuum never
      imports the Headroom package or stores originals.

See ``gap-analysis/headroom-native-integration-plan.md``.
"""

from __future__ import annotations

from continuum.llm.headroom.client import CompressionStats, HeadroomClient

__all__ = ["CompressionStats", "HeadroomClient"]
