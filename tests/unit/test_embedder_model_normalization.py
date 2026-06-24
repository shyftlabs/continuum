"""Embedder model-name normalization for gateway vs. direct routing.

EMBEDDER_MODEL should be a single fixed value while only EMBEDDER_API_BASE is
toggled. The OpenAI embedder branch of MemoryConfig._build_embedder_config
normalizes the model name to the routing target:

  * api_base set (gateway) → the gateway expects "<provider>/<model>"
  * no api_base (direct OpenAI) → OpenAI rejects a prefix, so use the bare name

Either stored form ("text-embedding-3-small" or "openai/text-embedding-3-small")
must resolve correctly in both modes. The direct case is the regression guard
for the `400 invalid model parameter` seen when a bare name was sent to the
gateway (and vice-versa).
"""

from __future__ import annotations

import pytest

from continuum.memory.config import MemoryConfig

GATEWAY = "https://gateway.example/v1"
BARE = "text-embedding-3-small"
PREFIXED = "openai/text-embedding-3-small"


def _embedder_model(*, api_base: str | None, model: str) -> tuple[str, str | None]:
    cfg = MemoryConfig(
        embedder_provider="openai",
        embedder_model=model,
        embedder_api_base=api_base,
    )
    provider, conf = cfg._build_embedder_config()
    assert provider == "openai"
    return conf["model"], conf.get("openai_base_url")


@pytest.mark.parametrize("stored", [BARE, PREFIXED])
def test_direct_uses_bare_model_and_no_base(stored):
    # No api_base → direct OpenAI: prefix stripped, no base URL set.
    model, base = _embedder_model(api_base=None, model=stored)
    assert model == BARE
    assert base is None


@pytest.mark.parametrize("stored", [BARE, PREFIXED])
def test_gateway_uses_prefixed_model_and_sets_base(stored):
    # api_base set → gateway: model carries the "<provider>/<model>" namespace.
    model, base = _embedder_model(api_base=GATEWAY, model=stored)
    assert model == PREFIXED
    assert base == GATEWAY


def test_single_fixed_model_works_in_both_modes():
    # The whole point: one fixed EMBEDDER_MODEL, only EMBEDDER_API_BASE toggled.
    fixed = BARE
    direct_model, _ = _embedder_model(api_base=None, model=fixed)
    gateway_model, _ = _embedder_model(api_base=GATEWAY, model=fixed)
    assert direct_model == BARE
    assert gateway_model == PREFIXED


def test_non_default_model_is_namespaced_for_gateway():
    # Normalization is not hard-coded to one model id.
    model, base = _embedder_model(api_base=GATEWAY, model="text-embedding-3-large")
    assert model == "openai/text-embedding-3-large"
    assert base == GATEWAY
