"""
Integration tests for Milvus Vector Database — real Milvus on port 19530.

Tests collection operations, health check, and vector storage.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


class TestMilvusIntegration:
    def test_milvus_is_healthy(self, real_milvus):
        """Milvus should respond to list_collections."""
        collections = real_milvus.list_collections()
        assert collections is not None

    def test_create_and_delete_collection(self, real_milvus, test_id):
        """Test creating and deleting a vector collection."""
        collection_name = f"test_collection_{test_id}".replace("-", "_")

        real_milvus.create_collection(
            collection_name=collection_name,
            dimension=4,
            metric_type="COSINE",
        )

        # Verify it exists
        collections = real_milvus.list_collections()
        assert collection_name in collections

        # Cleanup
        real_milvus.drop_collection(collection_name)

    def test_upsert_and_search_vectors(self, real_milvus, test_id):
        """Test real vector upsert and search."""
        collection_name = f"test_vectors_{test_id}".replace("-", "_")

        real_milvus.create_collection(
            collection_name=collection_name,
            dimension=4,
            metric_type="COSINE",
        )

        # Insert points
        real_milvus.insert(
            collection_name=collection_name,
            data=[
                {"id": 1, "vector": [0.1, 0.2, 0.3, 0.4], "text": "hello"},
                {"id": 2, "vector": [0.5, 0.6, 0.7, 0.8], "text": "world"},
            ],
        )
        real_milvus.flush(collection_name=collection_name)

        # Search
        results = real_milvus.search(
            collection_name=collection_name,
            data=[[0.1, 0.2, 0.3, 0.4]],
            limit=2,
            output_fields=["text"],
        )
        assert len(results[0]) == 2
        # Closest should be id 1
        assert results[0][0]["id"] == 1

        # Cleanup
        real_milvus.drop_collection(collection_name)
