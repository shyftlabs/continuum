"""
Integration tests for Qdrant Vector Database — real Qdrant on port 6333.

Tests collection operations, health check, and vector storage.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


class TestQdrantIntegration:
    def test_qdrant_is_healthy(self, real_qdrant):
        """Qdrant should respond to collection list."""
        collections = real_qdrant.get_collections()
        assert collections is not None

    def test_create_and_delete_collection(self, real_qdrant, test_id):
        """Test creating and deleting a vector collection."""
        from qdrant_client.models import Distance, VectorParams

        collection_name = f"test-collection-{test_id}"

        real_qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=4, distance=Distance.COSINE),
        )

        # Verify it exists
        collections = real_qdrant.get_collections()
        names = [c.name for c in collections.collections]
        assert collection_name in names

        # Cleanup
        real_qdrant.delete_collection(collection_name)

    def test_upsert_and_search_vectors(self, real_qdrant, test_id):
        """Test real vector upsert and search."""
        from qdrant_client.models import Distance, PointStruct, VectorParams

        collection_name = f"test-vectors-{test_id}"

        real_qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=4, distance=Distance.COSINE),
        )

        # Upsert points
        real_qdrant.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(id=1, vector=[0.1, 0.2, 0.3, 0.4], payload={"text": "hello"}),
                PointStruct(id=2, vector=[0.5, 0.6, 0.7, 0.8], payload={"text": "world"}),
            ],
        )

        # Search
        results = real_qdrant.query_points(
            collection_name=collection_name,
            query=[0.1, 0.2, 0.3, 0.4],
            limit=2,
        )
        assert len(results.points) == 2
        # Closest should be point 1
        assert results.points[0].id == 1

        # Cleanup
        real_qdrant.delete_collection(collection_name)
