"""
Type definitions for the Memory module.

Provides Pydantic models for structured data handling with memory operations.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

PROVENANCE_LABELS_KEY = "_data_labels"
"""Metadata key carrying a row's provenance labels (security finding F6).

The taint labels active on the run that produced a memory, stamped on write and
read back to re-taint any run that recalls the row. Underscore-prefixed to match
the internal metadata convention (``_user_id``, ``_agent_id``) and to stay out of
the way of integrator-supplied keys.

Stored as a sorted ``list[str]``: metadata round-trips through the vector store
as JSON, where a ``set`` is not serialisable and its iteration order is not
stable. Sorting keeps a row's stamp byte-identical for the same label set.

Why row-level rather than only scope-level: ``scope_data_labels`` taints on the
scope being read, which cannot separate a preference the user genuinely stated
from a sentence an attacker planted upstream and the model echoed -- both land as
rows in the same scope. The row's own provenance can.
"""


REVIEWED_KEY = "_reviewed"
"""Metadata key recording that a person reviewed a tainted row (finding F6).

Written by :meth:`MemoryClient.mark_reviewed` in place of
:data:`PROVENANCE_LABELS_KEY`, as ``{"by": ..., "at": ..., "cleared": [...]}``.

A record rather than an erasure. Provenance is deliberately coarse -- every row
a tainted run writes is stamped, so a useful fact learned from a fetched page
carries the same label as a planted instruction -- which means a real deployment
accumulates labelled rows that are genuinely fine. Clearing the label outright
would leave those indistinguishable from rows that were never tainted, so nobody
could later ask which a human had actually blessed, or who blessed them. Same
reasoning as the tool-catalogue approval in F3: the decision is the artifact.

The read path keys off ``PROVENANCE_LABELS_KEY`` alone, so a reviewed row renders
in the plain profile block and no longer taints the runs that recall it.
"""


class MemoryMetadata(BaseModel):
    """Metadata for a memory entry."""

    category: str | None = None
    tags: list[str] = Field(default_factory=list)
    source: str | None = None
    confidence: float | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    custom: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for mem0."""
        data: dict[str, Any] = {}

        if self.category:
            data["category"] = self.category
        if self.tags:
            data["tags"] = self.tags
        if self.source:
            data["source"] = self.source
        if self.confidence is not None:
            data["confidence"] = self.confidence
        if self.created_at:
            data["created_at"] = self.created_at.isoformat()
        if self.updated_at:
            data["updated_at"] = self.updated_at.isoformat()

        # Merge custom metadata
        if self.custom:
            data.update(self.custom)

        return data


class MemoryEntry(BaseModel):
    """Represents a single memory entry."""

    id: str
    memory: str
    hash: str | None = None
    user_id: str | None = None
    agent_id: str | None = None
    conversation_id: str | None = None  # maps to mem0's run_id internally
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    score: float | None = None  # Relevance score from search

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        data: dict[str, Any] = {
            "id": self.id,
            "memory": self.memory,
        }

        if self.hash:
            data["hash"] = self.hash
        if self.user_id:
            data["user_id"] = self.user_id
        if self.agent_id:
            data["agent_id"] = self.agent_id
        if self.conversation_id:
            data["conversation_id"] = self.conversation_id
        if self.metadata:
            data["metadata"] = self.metadata
        if self.created_at:
            data["created_at"] = (
                self.created_at.isoformat()
                if isinstance(self.created_at, datetime)
                else self.created_at
            )
        if self.updated_at:
            data["updated_at"] = (
                self.updated_at.isoformat()
                if isinstance(self.updated_at, datetime)
                else self.updated_at
            )
        if self.score is not None:
            data["score"] = self.score

        return data

    @classmethod
    def from_mem0_result(cls, result: dict[str, Any]) -> "MemoryEntry":
        """Create MemoryEntry from mem0 result."""
        # Handle metadata - ensure it's always a dict
        metadata = result.get("metadata")
        if metadata is None:
            metadata = {}
        elif not isinstance(metadata, dict):
            metadata = {}

        return cls(
            id=result.get("id", ""),
            memory=result.get("memory", ""),
            hash=result.get("hash"),
            user_id=result.get("user_id"),
            agent_id=result.get("agent_id"),
            conversation_id=result.get("run_id"),  # mem0 stores conversation_id as run_id
            metadata=metadata,
            created_at=result.get("created_at"),
            updated_at=result.get("updated_at"),
            score=result.get("score"),
        )


class MemorySearchResult(BaseModel):
    """Results from a memory search operation."""

    results: list[MemoryEntry]
    query: str
    limit: int
    total_results: int | None = None

    @classmethod
    def from_mem0_response(
        cls, response: dict[str, Any], query: str, limit: int
    ) -> "MemorySearchResult":
        """Create MemorySearchResult from mem0 search response."""
        results_data = response.get("results", [])
        entries = [MemoryEntry.from_mem0_result(r) for r in results_data]

        return cls(
            results=entries,
            query=query,
            limit=limit,
            total_results=len(entries),
        )

    def get_memory_strings(self) -> list[str]:
        """Get just the memory text from all results."""
        return [entry.memory for entry in self.results]

    def get_top_k(self, k: int) -> list[MemoryEntry]:
        """Get top K results by score."""
        sorted_results = sorted(
            self.results, key=lambda x: x.score if x.score is not None else 0.0, reverse=True
        )
        return sorted_results[:k]


class MemoryAddResult(BaseModel):
    """Result from adding memories."""

    message: str
    results: list[dict[str, Any]] = Field(default_factory=list)
    relations: list[dict[str, Any]] = Field(default_factory=list)

    # Facts a pre_store_filter stopped before the write. They are NOT in the
    # store, but mem0's ADD branch still returns them (it appends the result
    # entry whatever _create_memory returned), so `results` alone would report a
    # suppressed fact as stored.
    suppressed: list[str] = Field(default_factory=list)

    @classmethod
    def from_mem0_response(cls, response: dict[str, Any] | str) -> "MemoryAddResult":
        """Create MemoryAddResult from mem0 add response."""
        if isinstance(response, str):
            return cls(message=response)

        return cls(
            message=response.get("message", "Memory added successfully"),
            results=response.get("results", []),
            relations=response.get("relations", []),
        )


class MemoryFilter(BaseModel):
    """Filter criteria for memory operations."""

    user_id: str | None = None
    agent_id: str | None = None
    conversation_id: str | None = None
    category: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_mem0_filter(self) -> dict[str, Any]:
        """Convert to mem0 filter format."""
        filter_dict: dict[str, Any] = {}

        if self.user_id:
            filter_dict["user_id"] = self.user_id
        if self.agent_id:
            filter_dict["agent_id"] = self.agent_id
        if self.conversation_id:
            filter_dict["run_id"] = self.conversation_id  # mem0 uses run_id internally
        if self.category:
            filter_dict["category"] = self.category
        if self.tags:
            filter_dict["tags"] = self.tags
        if self.metadata:
            filter_dict.update(self.metadata)

        return filter_dict
