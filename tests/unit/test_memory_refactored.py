"""
Tests for the refactored memory module.

Tests the new provider-based architecture with:
- BaseMemoryProvider interface
- Mem0Provider implementation
- MemoryScope management
- MemoryClient delegation
"""

import logging

import pytest

from orchestrator.memory import (
    BaseMemoryProvider,
    MemoryAddResult,
    MemoryClient,
    MemoryConfig,
    MemoryEntry,
    MemoryFilter,
    MemoryIdentifierError,
    MemoryMetadata,
    MemoryNotEnabledError,
    MemoryScope,
    MemorySearchResult,
)
from orchestrator.memory.providers import (
    create_provider,
    get_provider_class,
    is_mem0_available,
    list_providers,
    register_provider,
)

logger = logging.getLogger(__name__)


# =============================================================================
# MemoryScope Tests
# =============================================================================


class TestMemoryScope:
    """Tests for MemoryScope class."""

    def test_shared_scope(self):
        """Test creating shared scope."""
        logger.info("Test creating shared scope")
        scope = MemoryScope.shared()
        assert scope.agent_id == "shared"
        assert scope.user_id is None
        assert scope.conversation_id is None
        assert scope.to_identifiers() == {"agent_id": "shared"}

    def test_user_scope(self):
        """Test creating user scope."""
        logger.info("Test creating user scope")
        scope = MemoryScope.user("user-123")
        assert scope.user_id == "user-123"
        assert scope.agent_id is None
        assert scope.conversation_id is None
        assert scope.to_identifiers() == {"user_id": "user-123"}

    def test_agent_scope(self):
        """Test creating agent scope."""
        logger.info("Test creating agent scope")
        scope = MemoryScope.agent("agent-456")
        assert scope.agent_id == "agent-456"
        assert scope.user_id is None
        assert scope.conversation_id is None
        assert scope.to_identifiers() == {"agent_id": "agent-456"}

    def test_conversation_scope(self):
        """Test creating conversation scope."""
        logger.info("Test creating conversation scope")
        scope = MemoryScope.conversation("conv-789")
        assert scope.conversation_id == "conv-789"
        assert scope.user_id is None
        assert scope.agent_id is None
        assert scope.to_identifiers() == {"conversation_id": "conv-789"}

    def test_from_isolation_mode_shared(self):
        """Test creating scope from shared isolation mode."""
        logger.info("Test creating scope from shared isolation mode")
        scope = MemoryScope.from_isolation_mode("shared")
        assert scope.agent_id == "shared"

    def test_from_isolation_mode_user(self):
        """Test creating scope from user isolation mode."""
        logger.info("Test creating scope from user isolation mode")
        scope = MemoryScope.from_isolation_mode(
            "user",
            user_id="user-123",
            agent_id="agent-456",
            conversation_id="conv-789",
        )
        assert scope.user_id == "user-123"
        # Other identifiers are not set in the scope
        assert scope.agent_id is None
        assert scope.conversation_id is None

    def test_from_isolation_mode_missing_identifier(self):
        """Test error when required identifier is missing."""
        logger.info("Test error when required identifier is missing")
        with pytest.raises(ValueError, match="user_id.*required"):
            MemoryScope.from_isolation_mode("user")

        with pytest.raises(ValueError, match="agent_id.*required"):
            MemoryScope.from_isolation_mode("agent")

        with pytest.raises(ValueError, match="conversation_id.*required"):
            MemoryScope.from_isolation_mode("conversation")

    def test_from_identifiers(self):
        """Test creating scope from explicit identifiers."""
        logger.info("Test creating scope from explicit identifiers")
        scope = MemoryScope.from_identifiers(
            user_id="user-123",
            agent_id="agent-456",
            conversation_id="conv-789",
        )
        assert scope.user_id == "user-123"
        assert scope.agent_id == "agent-456"
        assert scope.conversation_id == "conv-789"
        assert scope.to_identifiers() == {
            "user_id": "user-123",
            "agent_id": "agent-456",
            "conversation_id": "conv-789",
        }

    def test_to_metadata(self):
        """Test scope to metadata conversion."""
        logger.info("Test scope to metadata conversion")
        scope = MemoryScope(user_id="user-123", agent_id="agent-456")
        metadata = scope.to_metadata()
        assert metadata == {
            "_user_id": "user-123",
            "_agent_id": "agent-456",
        }

    def test_validate_for_mode(self):
        """Test scope validation for specific modes."""
        logger.info("Test scope validation for specific modes")
        scope = MemoryScope(user_id="user-123")

        # Valid for user mode
        is_valid, error = scope.validate_for_mode("user")
        assert is_valid
        assert error is None

        # Invalid for agent mode
        is_valid, error = scope.validate_for_mode("agent")
        assert not is_valid
        assert "agent_id" in error and "required" in error

    def test_is_empty(self):
        """Test empty scope detection."""
        logger.info("Test empty scope detection")
        empty_scope = MemoryScope()
        assert empty_scope.is_empty()

        non_empty = MemoryScope(user_id="user-123")
        assert not non_empty.is_empty()

    def test_repr(self):
        """Test string representation."""
        logger.info("Test string representation")
        scope = MemoryScope(user_id="user-123")
        assert "user_id='user-123'" in repr(scope)

        empty_scope = MemoryScope()
        assert "empty" in repr(empty_scope)


# =============================================================================
# MemoryConfig Tests
# =============================================================================


class TestMemoryConfig:
    """Tests for MemoryConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        logger.info("Test default configuration")
        config = MemoryConfig(enabled=False)
        # vector_store_provider defaults from settings (environment-dependent),
        # so just verify it's one of the valid options
        assert config.vector_store_provider in ("qdrant", "milvus")
        assert config.memory_isolation in ["shared", "user", "agent", "conversation"]

    def test_is_configured_disabled(self):
        """Test is_configured when disabled."""
        logger.info("Test is_configured when disabled")
        config = MemoryConfig(enabled=False)
        assert not config.is_configured()

    def test_is_configured_missing_fields(self):
        """Test is_configured with missing fields."""
        logger.info("Test is_configured with missing fields")
        config = MemoryConfig(
            enabled=True,
            vector_store_provider="qdrant",
            qdrant_host="",
            memory_llm_model="gpt-4o-mini",
            embedder_model="text-embedding-3-small",
            embedding_dims=1536,
        )
        assert not config.is_configured()

    def test_is_configured_complete(self):
        """Test is_configured with complete config."""
        logger.info("Test is_configured with complete config")
        config = MemoryConfig(
            enabled=True,
            qdrant_host="localhost",
            memory_llm_model="gpt-4o-mini",
            embedder_model="text-embedding-3-small",
            embedding_dims=1536,
        )
        assert config.is_configured()

    def test_to_mem0_config(self):
        """Test conversion to mem0 config format."""
        logger.info("Test conversion to mem0 config format")
        config = MemoryConfig(
            enabled=True,
            vector_store_provider="qdrant",
            qdrant_host="localhost",
            qdrant_port=6333,
            qdrant_collection="test_collection",
            memory_llm_model="gpt-4o-mini",
            memory_llm_temperature=0.1,
            embedder_provider="openai",
            embedder_model="text-embedding-3-small",
            embedding_dims=1536,
            history_db_path="/tmp/test_history.db",
        )

        mem0_config = config.to_mem0_config()

        assert mem0_config["version"] == "v1.1"
        assert mem0_config["llm"]["provider"] == "openai"
        assert mem0_config["llm"]["config"]["model"] == "gpt-4o-mini"
        assert mem0_config["embedder"]["provider"] == "openai"
        assert mem0_config["embedder"]["config"]["model"] == "text-embedding-3-small"
        assert mem0_config["embedder"]["config"]["embedding_dims"] == 1536
        assert mem0_config["vector_store"]["provider"] == "qdrant"
        assert mem0_config["vector_store"]["config"]["host"] == "localhost"
        assert mem0_config["vector_store"]["config"]["port"] == 6333

    def test_is_configured_missing_fields_milvus(self):
        """Test is_configured with missing milvus fields."""
        logger.info("Test is_configured with missing milvus fields")
        config = MemoryConfig(
            enabled=True,
            vector_store_provider="milvus",
            milvus_host="",
            memory_llm_model="gpt-4o-mini",
            embedder_model="text-embedding-3-small",
            embedding_dims=1536,
        )
        assert not config.is_configured()

    def test_is_configured_complete_milvus(self):
        """Test is_configured with complete milvus config."""
        logger.info("Test is_configured with complete milvus config")
        config = MemoryConfig(
            enabled=True,
            vector_store_provider="milvus",
            milvus_host="localhost",
            milvus_port=19530,
            memory_llm_model="gpt-4o-mini",
            embedder_model="text-embedding-3-small",
            embedding_dims=1536,
        )
        assert config.is_configured()

    def test_to_mem0_config_milvus(self):
        """Test conversion to mem0 config format with milvus."""
        logger.info("Test conversion to mem0 config format with milvus")
        config = MemoryConfig(
            enabled=True,
            vector_store_provider="milvus",
            milvus_host="localhost",
            milvus_port=19530,
            milvus_collection="test_collection",
            memory_llm_model="gpt-4o-mini",
            memory_llm_temperature=0.1,
            embedder_provider="openai",
            embedder_model="text-embedding-3-small",
            embedding_dims=1536,
            history_db_path="/tmp/test_history.db",
        )

        mem0_config = config.to_mem0_config()

        assert mem0_config["version"] == "v1.1"
        assert mem0_config["llm"]["provider"] == "openai"
        assert mem0_config["llm"]["config"]["model"] == "gpt-4o-mini"
        assert mem0_config["embedder"]["provider"] == "openai"
        assert mem0_config["embedder"]["config"]["model"] == "text-embedding-3-small"
        assert mem0_config["embedder"]["config"]["embedding_dims"] == 1536
        assert mem0_config["vector_store"]["provider"] == "milvus"
        assert mem0_config["vector_store"]["config"]["collection_name"] == "test_collection"
        assert mem0_config["vector_store"]["config"]["embedding_model_dims"] == 1536
        assert mem0_config["vector_store"]["config"]["url"] == "http://localhost:19530"

    def test_to_mem0_config_milvus_with_token(self):
        """Test milvus config includes token when provided (e.g. Zilliz Cloud)."""
        logger.info("Test milvus config includes token when provided")
        config = MemoryConfig(
            enabled=True,
            vector_store_provider="milvus",
            milvus_host="my-instance.zillizcloud.com",
            milvus_port=19530,
            milvus_token="my-zilliz-token",
            milvus_collection="test_collection",
            memory_llm_model="gpt-4o-mini",
            embedder_model="text-embedding-3-small",
            embedding_dims=1536,
            history_db_path="/tmp/test_history.db",
        )

        mem0_config = config.to_mem0_config()

        assert mem0_config["vector_store"]["provider"] == "milvus"
        assert mem0_config["vector_store"]["config"]["token"] == "my-zilliz-token"


# =============================================================================
# Mock Provider for Testing
# =============================================================================


class MockProvider(BaseMemoryProvider):
    """Mock memory provider for testing."""

    def __init__(self, config: MemoryConfig | None = None):
        self._config = config
        self._initialized = True
        self._memories: dict[str, dict] = {}

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def add(
        self,
        messages,
        *,
        user_id=None,
        agent_id=None,
        run_id=None,
        metadata=None,
        custom_prompt=None,
        infer=True,
    ):
        return MemoryAddResult(message="Added", results=[{"id": "test-1"}])

    async def search(
        self, query, *, user_id=None, agent_id=None, run_id=None, limit=5, filters=None
    ):
        return MemorySearchResult(
            results=[MemoryEntry(id="test-1", memory="Test memory", score=0.9)],
            query=query,
            limit=limit,
            total_results=1,
        )

    async def get(self, memory_id):
        return MemoryEntry(id=memory_id, memory="Test memory")

    async def get_all(self, *, user_id=None, agent_id=None, run_id=None, limit=None):
        return [MemoryEntry(id="test-1", memory="Test memory")]

    async def delete(self, memory_id):
        return True

    async def delete_all(self, *, user_id=None, agent_id=None, run_id=None):
        return True

    async def update(self, memory_id, data, *, custom_prompt=None):
        return MemoryEntry(id=memory_id, memory=data)

    async def history(self, memory_id):
        return [{"version": 1, "memory": "Test memory"}]

    async def reset(self):
        return True

    async def close(self):
        self._initialized = False

    # Sync methods
    def add_sync(self, messages, **kwargs):
        return MemoryAddResult(message="Added", results=[{"id": "test-1"}])

    def search_sync(self, query, **kwargs):
        return MemorySearchResult(
            results=[MemoryEntry(id="test-1", memory="Test memory", score=0.9)],
            query=query,
            limit=kwargs.get("limit", 5),
            total_results=1,
        )

    def get_sync(self, memory_id):
        return MemoryEntry(id=memory_id, memory="Test memory")

    def get_all_sync(self, **kwargs):
        return [MemoryEntry(id="test-1", memory="Test memory")]

    def delete_sync(self, memory_id):
        return True

    def delete_all_sync(self, **kwargs):
        return True

    def update_sync(self, memory_id, data, **kwargs):
        return MemoryEntry(id=memory_id, memory=data)

    def history_sync(self, memory_id):
        return [{"version": 1, "memory": "Test memory"}]

    def reset_sync(self):
        return True


# =============================================================================
# MemoryClient Tests
# =============================================================================


class TestMemoryClient:
    """Tests for MemoryClient class."""

    def test_init_disabled(self):
        """Test client initialization when disabled."""
        logger.info("Test client initialization when disabled")
        config = MemoryConfig(enabled=False)
        client = MemoryClient(config=config)

        assert not client.is_enabled
        assert client.provider is None

    def test_init_with_provider(self):
        """Test client initialization with explicit provider."""
        logger.info("Test client initialization with explicit provider")
        config = MemoryConfig(enabled=True)
        provider = MockProvider(config)
        client = MemoryClient(config=config, provider=provider)

        assert client.is_enabled
        assert client.provider is provider

    def test_ensure_enabled_raises(self):
        """Test that operations raise when not enabled."""
        logger.info("Test that operations raise when not enabled")
        config = MemoryConfig(enabled=False)
        client = MemoryClient(config=config)

        with pytest.raises(MemoryNotEnabledError):
            client._ensure_enabled()

    def test_build_scope_user_mode(self):
        """Test scope building in user mode."""
        logger.info("Test scope building in user mode")
        config = MemoryConfig(enabled=True, memory_isolation="user")
        provider = MockProvider(config)
        client = MemoryClient(config=config, provider=provider)

        scope = client._build_scope(user_id="user-123")
        assert scope.user_id == "user-123"

    def test_build_scope_missing_identifier(self):
        """Test scope building with missing required identifier."""
        logger.info("Test scope building with missing required identifier")
        config = MemoryConfig(enabled=True, memory_isolation="user")
        provider = MockProvider(config)
        client = MemoryClient(config=config, provider=provider)

        with pytest.raises(MemoryIdentifierError):
            client._build_scope()  # Missing user_id for user mode

    @pytest.mark.asyncio
    async def test_add(self):
        """Test async add method."""
        logger.info("Test async add method")
        config = MemoryConfig(enabled=True, memory_isolation="user")
        provider = MockProvider(config)
        client = MemoryClient(config=config, provider=provider)

        result = await client.add(
            "Test message",
            user_id="user-123",
            metadata={"category": "test"},
        )

        assert result.message == "Added"
        assert len(result.results) == 1

    @pytest.mark.asyncio
    async def test_search(self):
        """Test async search method."""
        logger.info("Test async search method")
        config = MemoryConfig(enabled=True, memory_isolation="user")
        provider = MockProvider(config)
        client = MemoryClient(config=config, provider=provider)

        result = await client.search(
            "test query",
            user_id="user-123",
            limit=5,
        )

        assert result.total_results == 1
        assert result.results[0].memory == "Test memory"

    @pytest.mark.asyncio
    async def test_get(self):
        """Test async get method."""
        logger.info("Test async get method")
        config = MemoryConfig(enabled=True, memory_isolation="user")
        provider = MockProvider(config)
        client = MemoryClient(config=config, provider=provider)

        result = await client.get("test-1")

        assert result is not None
        assert result.id == "test-1"

    @pytest.mark.asyncio
    async def test_delete(self):
        """Test async delete method."""
        logger.info("Test async delete method")
        config = MemoryConfig(enabled=True, memory_isolation="user")
        provider = MockProvider(config)
        client = MemoryClient(config=config, provider=provider)

        result = await client.delete("test-1")
        assert result is True

    @pytest.mark.asyncio
    async def test_update(self):
        """Test async update method."""
        logger.info("Test async update method")
        config = MemoryConfig(enabled=True, memory_isolation="user")
        provider = MockProvider(config)
        client = MemoryClient(config=config, provider=provider)

        result = await client.update("test-1", "Updated memory")

        assert result.id == "test-1"
        assert result.memory == "Updated memory"

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        logger.info("Test async context manager")
        config = MemoryConfig(enabled=True, memory_isolation="user")
        provider = MockProvider(config)

        async with MemoryClient(config=config, provider=provider) as client:
            assert client.is_enabled

        # Provider should be closed
        assert not provider.is_initialized

    def test_sync_search(self):
        """Test sync search method."""
        logger.info("Test sync search method")
        config = MemoryConfig(enabled=True, memory_isolation="user")
        provider = MockProvider(config)
        client = MemoryClient(config=config, provider=provider)

        result = client.search_sync(
            "test query",
            user_id="user-123",
        )

        assert result.total_results == 1

    def test_sync_add(self):
        """Test sync add method."""
        logger.info("Test sync add method")
        config = MemoryConfig(enabled=True, memory_isolation="user")
        provider = MockProvider(config)
        client = MemoryClient(config=config, provider=provider)

        result = client.add_sync(
            "Test message",
            user_id="user-123",
        )

        assert result.message == "Added"


# =============================================================================
# Provider Registry Tests
# =============================================================================


class TestProviderRegistry:
    """Tests for provider registry functions."""

    def test_list_providers(self):
        """Test listing available providers."""
        logger.info("Test listing available providers")
        providers = list_providers()
        # mem0 should be registered by default if mem0ai is installed
        assert isinstance(providers, list)
        if is_mem0_available():
            assert "mem0" in providers

    def test_is_mem0_available(self):
        """Test mem0 availability check."""
        # This should return a boolean
        logger.info("Test mem0 availability check")
        available = is_mem0_available()
        assert isinstance(available, bool)

    def test_register_provider(self):
        """Test registering a custom provider."""
        logger.info("Test registering a custom provider")
        register_provider("mock", MockProvider)

        providers = list_providers()
        assert "mock" in providers

    def test_get_provider_class(self):
        """Test getting a provider class."""
        logger.info("Test getting a provider class")
        register_provider("mock", MockProvider)

        provider_class = get_provider_class("mock")
        assert provider_class is MockProvider

    def test_get_provider_class_not_found(self):
        """Test error when provider not found."""
        logger.info("Test error when provider not found")
        with pytest.raises(ValueError, match="Unknown memory provider"):
            get_provider_class("nonexistent")

    def test_create_provider(self):
        """Test creating a provider instance."""
        logger.info("Test creating a provider instance")
        register_provider("mock", MockProvider)
        config = MemoryConfig(enabled=True)

        provider = create_provider("mock", config)

        assert isinstance(provider, MockProvider)
        assert provider.provider_name == "mock"


# =============================================================================
# Types Tests
# =============================================================================


class TestMemoryTypes:
    """Tests for memory type classes."""

    def test_memory_entry_from_mem0_result(self):
        """Test creating MemoryEntry from mem0 result."""
        logger.info("Test creating MemoryEntry from mem0 result")
        result = {
            "id": "test-1",
            "memory": "Test memory content",
            "hash": "abc123",
            "user_id": "user-123",
            "metadata": {"category": "test"},
            "score": 0.95,
        }

        entry = MemoryEntry.from_mem0_result(result)

        assert entry.id == "test-1"
        assert entry.memory == "Test memory content"
        assert entry.hash == "abc123"
        assert entry.user_id == "user-123"
        assert entry.metadata == {"category": "test"}
        assert entry.score == 0.95

    def test_memory_entry_with_none_metadata(self):
        """Test MemoryEntry handles None metadata."""
        logger.info("Test MemoryEntry handles None metadata")
        result = {
            "id": "test-1",
            "memory": "Test",
            "metadata": None,
        }

        entry = MemoryEntry.from_mem0_result(result)
        assert entry.metadata == {}

    def test_memory_search_result_from_mem0(self):
        """Test creating MemorySearchResult from mem0 response."""
        logger.info("Test creating MemorySearchResult from mem0 response")
        response = {
            "results": [
                {"id": "1", "memory": "Memory 1", "score": 0.9},
                {"id": "2", "memory": "Memory 2", "score": 0.8},
            ]
        }

        result = MemorySearchResult.from_mem0_response(response, "query", 5)

        assert result.total_results == 2
        assert result.query == "query"
        assert result.limit == 5
        assert len(result.results) == 2

    def test_memory_search_result_get_top_k(self):
        """Test getting top K results."""
        logger.info("Test getting top K results")
        result = MemorySearchResult(
            results=[
                MemoryEntry(id="1", memory="Low", score=0.5),
                MemoryEntry(id="2", memory="High", score=0.9),
                MemoryEntry(id="3", memory="Medium", score=0.7),
            ],
            query="test",
            limit=10,
            total_results=3,
        )

        top_2 = result.get_top_k(2)

        assert len(top_2) == 2
        assert top_2[0].id == "2"  # Highest score
        assert top_2[1].id == "3"  # Second highest

    def test_memory_add_result_from_string(self):
        """Test creating MemoryAddResult from string response."""
        logger.info("Test creating MemoryAddResult from string response")
        result = MemoryAddResult.from_mem0_response("Memory added")
        assert result.message == "Memory added"

    def test_memory_add_result_from_dict(self):
        """Test creating MemoryAddResult from dict response."""
        logger.info("Test creating MemoryAddResult from dict response")
        response = {
            "message": "Added successfully",
            "results": [{"id": "1"}],
            "relations": [],
        }

        result = MemoryAddResult.from_mem0_response(response)

        assert result.message == "Added successfully"
        assert len(result.results) == 1

    def test_memory_metadata_to_dict(self):
        """Test MemoryMetadata to dict conversion."""
        logger.info("Test MemoryMetadata to dict conversion")
        metadata = MemoryMetadata(
            category="preferences",
            tags=["important", "user"],
            source="conversation",
            custom={"key": "value"},
        )

        result = metadata.to_dict()

        assert result["category"] == "preferences"
        assert result["tags"] == ["important", "user"]
        assert result["source"] == "conversation"
        assert result["key"] == "value"  # Custom merged

    def test_memory_filter_to_mem0_filter(self):
        """Test MemoryFilter to mem0 format conversion."""
        logger.info("Test MemoryFilter to mem0 format conversion")
        filter_obj = MemoryFilter(
            user_id="user-123",
            category="test",
            tags=["tag1", "tag2"],
            metadata={"custom": "value"},
        )

        result = filter_obj.to_mem0_filter()

        assert result["user_id"] == "user-123"
        assert result["category"] == "test"
        assert result["tags"] == ["tag1", "tag2"]
        assert result["custom"] == "value"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the memory module."""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test a full memory workflow."""
        logger.info("Test a full memory workflow")
        config = MemoryConfig(enabled=True, memory_isolation="user")
        provider = MockProvider(config)

        async with MemoryClient(config=config, provider=provider) as client:
            # Add a memory
            add_result = await client.add(
                "User prefers dark mode",
                user_id="user-123",
                metadata={"category": "preferences"},
            )
            assert add_result.message == "Added"

            # Search for memories
            search_result = await client.search(
                "What are user preferences?",
                user_id="user-123",
                limit=5,
            )
            assert search_result.total_results >= 1

            # Get a specific memory
            memory = await client.get("test-1")
            assert memory is not None

            # Update the memory
            updated = await client.update("test-1", "User prefers light mode")
            assert updated.memory == "User prefers light mode"

            # Delete the memory
            deleted = await client.delete("test-1")
            assert deleted

    def test_isolation_modes(self):
        """Test different isolation modes."""
        logger.info("Test different isolation modes")
        for mode in ["shared", "user", "agent", "conversation"]:
            config = MemoryConfig(enabled=True, memory_isolation=mode)
            provider = MockProvider(config)
            client = MemoryClient(config=config, provider=provider)

            # Build appropriate scope
            kwargs = {}
            if mode == "user":
                kwargs["user_id"] = "user-123"
            elif mode == "agent":
                kwargs["agent_id"] = "agent-456"
            elif mode == "conversation":
                kwargs["conversation_id"] = "conv-789"

            scope = client._build_scope(**kwargs)
            identifiers = scope.to_identifiers()

            # Verify correct identifier is set
            if mode == "shared":
                assert identifiers.get("agent_id") == "shared"
            elif mode == "user":
                assert "user_id" in identifiers
            elif mode == "agent":
                assert "agent_id" in identifiers
            elif mode == "conversation":
                assert "conversation_id" in identifiers
