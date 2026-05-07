"""
Memory module - Long-term memory with provider abstraction.

This module provides long-term memory capabilities for agents with a
provider-based architecture. Currently supports mem0 with Qdrant.

Architecture:
    - BaseMemoryProvider: Abstract interface for memory providers
    - Mem0Provider: Implementation using mem0's AsyncMemory and Memory
    - MemoryClient: High-level client that delegates to providers
    - MemoryScope: Scope management for memory isolation

Features:
    - Multi-level memory scoping (user, agent, run, shared)
    - Both async and sync interfaces
    - Semantic search with vector similarity
    - Automatic fact extraction and consolidation
    - Custom prompts for fact extraction and updates
    - Qdrant vector store integration
    - Full observability with Langfuse

Supported Embedder Providers (via mem0):
    - OpenAI (text-embedding-3-small, text-embedding-3-large)
    - Azure OpenAI
    - Hugging Face (BAAI/bge-m3, sentence-transformers/*)
    - Cohere (embed-english-v3.0, embed-multilingual-v3.0)
    - Google Gemini / Vertex AI
    - Ollama (local embeddings - nomic-embed-text, mxbai-embed-large)

Example:
    ```python
    from orchestrator.memory import MemoryClient, MemoryScope

    # Initialize client
    client = MemoryClient()

    # Add memories with user scope
    await client.add(
        "User prefers dark mode",
        user_id="user-123",
        metadata={"category": "preferences"}
    )

    # Search memories
    results = await client.search(
        "What are the user's preferences?",
        user_id="user-123",
        limit=5
    )

    # Use MemoryScope for explicit scope management
    scope = MemoryScope.user("user-123")
    scope = MemoryScope.from_isolation_mode("agent", agent_id="my-agent")
    ```

Adding New Providers:
    1. Create a new file in providers/ (e.g., pinecone.py)
    2. Implement BaseMemoryProvider interface
    3. Register in providers/__init__.py

See docs/memory.md for complete documentation.
"""

# Silence mem0's optional-extra startup warnings (spaCy/fastembed). They fire
# every time mem0 is imported and confuse hackathon participants who think the
# warnings indicate something they need to fix. Semantic vector search is
# unaffected by these missing optional extras. Participants who want
# hybrid/lemma search can `pip install "mem0ai[nlp]" fastembed`.
import logging as _logging

for _noisy in (
    "mem0.utils.spacy_models",
    "mem0.vector_stores.qdrant",
):
    _logging.getLogger(_noisy).setLevel(_logging.ERROR)


# Client
# Base class (for custom providers)
from orchestrator.memory.base import BaseMemoryProvider
from orchestrator.memory.client import (
    MemoryClient,
    get_global_memory_client,
    initialize_global_memory,
)

# Configuration
from orchestrator.memory.config import MemoryConfig

# Intelligence layer
from orchestrator.memory.intelligence import IntelligenceConfig, IntelligentMemoryClient

# Exceptions
from orchestrator.memory.exceptions import (
    MemoryAddError,
    MemoryConfigurationError,
    MemoryConnectionError,
    MemoryDeleteError,
    MemoryError,
    MemoryIdentifierError,
    MemoryNotEnabledError,
    MemorySearchError,
    MemoryUpdateError,
)

# Provider utilities
from orchestrator.memory.providers import (
    create_provider,
    get_provider_class,
    list_providers,
    register_provider,
)

# Scopes
from orchestrator.memory.scopes import (
    MemoryIsolationLevel,
    MemoryScope,
    ScopeDefinition,
    get_scope_definition,
    is_scope_registered,
    list_scopes,
    register_scope,
)

# Types
from orchestrator.memory.types import (
    MemoryAddResult,
    MemoryEntry,
    MemoryFilter,
    MemoryMetadata,
    MemorySearchResult,
)

__all__ = [
    # Client
    "MemoryClient",
    "get_global_memory_client",
    "initialize_global_memory",
    # Intelligence layer
    "IntelligentMemoryClient",
    "IntelligenceConfig",
    # Config
    "MemoryConfig",
    # Scopes
    "MemoryScope",
    "MemoryIsolationLevel",
    "ScopeDefinition",
    "register_scope",
    "get_scope_definition",
    "list_scopes",
    "is_scope_registered",
    # Types
    "MemoryEntry",
    "MemorySearchResult",
    "MemoryAddResult",
    "MemoryMetadata",
    "MemoryFilter",
    # Exceptions
    "MemoryError",
    "MemoryConfigurationError",
    "MemoryNotEnabledError",
    "MemoryConnectionError",
    "MemorySearchError",
    "MemoryAddError",
    "MemoryDeleteError",
    "MemoryUpdateError",
    "MemoryIdentifierError",
    # Base class (for custom providers)
    "BaseMemoryProvider",
    # Provider utilities
    "register_provider",
    "get_provider_class",
    "create_provider",
    "list_providers",
]
