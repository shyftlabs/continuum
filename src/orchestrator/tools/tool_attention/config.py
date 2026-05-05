"""Tool-attention configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolAttentionConfig:
    """
    Configuration for tool-attention routing.

    When attached to AgentConfig.tool_attention, the runner will route tool
    schemas semantically each turn instead of sending every schema every turn.

    Only activates when the agent has >= min_tools tools.
    """

    k: int = 5
    min_tools: int = 10
    threshold: float = 0.0
    always_promote: list[str] = field(default_factory=list)
    collection_name: str = "tool_attention_summaries"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
