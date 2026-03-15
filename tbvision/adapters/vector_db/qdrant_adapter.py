import uuid
from collections.abc import Sequence
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from qdrant_client.models import (
    FieldCondition,
    Filter,
    HasIdCondition,
    HasVectorCondition,
    IsEmptyCondition,
    IsNullCondition,
    MatchValue,
    NestedCondition,
)

from tbvision.adapters.vector_db.base import VectorDBAdapter


class QdrantAdapter(VectorDBAdapter):
    def __init__(self, url: str = "http://localhost:6333", vector_size: int = 384):
        self.client = AsyncQdrantClient(url=url)
        self.vector_size = vector_size
        self._collections_ready: set[str] = set()

    async def ensure_collection(self, collection_name: str) -> None:
        if collection_name in self._collections_ready:
            return
        collections = await self.client.get_collections()
        existing = {c.name for c in collections.collections}
        if collection_name not in existing:
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size, distance=Distance.COSINE
                ),
            )
        self._collections_ready.add(collection_name)

    async def store_embedding(
        self,
        id: str,
        embedding: list[float],
        metadata: dict[str, Any],
        collection_name: str,
    ):
        await self.ensure_collection(collection_name)
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={**metadata, "document_id": id},
        )
        await self.client.upsert(collection_name=collection_name, points=[point])

    async def upsert_points(self, points: Sequence[Any], collection_name: str) -> None:
        await self.ensure_collection(collection_name)
        await self.client.upsert(collection_name=collection_name, points=list(points))

    async def delete_by_document_id(self, document_id: str, collection_name: str):
        await self.ensure_collection(collection_name)
        await self.client.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id", match=MatchValue(value=document_id)
                    )
                ]
            ),
        )

    async def search_similar(
        self,
        query_embedding: list[float],
        top_k: int,
        collection_name: str,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        await self.ensure_collection(collection_name)

        qdrant_filter = None
        if filters:
            conditions: list[
                FieldCondition
                | IsEmptyCondition
                | IsNullCondition
                | HasIdCondition
                | NestedCondition
                | Filter
                | HasVectorCondition
            ] = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
                if v is not None
            ]
            if conditions:
                qdrant_filter = Filter(must=conditions)

        results = await self.client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=top_k,
            query_filter=qdrant_filter,
        )

        return [
            {
                "id": hit.id,
                "score": hit.score,
                "content": hit.payload.get("content") if hit.payload else None,
                "metadata": hit.payload,
            }
            for hit in results.points
        ]
