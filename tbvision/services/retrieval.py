from typing import Any

from tbvision.adapters.embeddings.base import EmbeddingAdapter
from tbvision.adapters.vector_db.base import VectorDBAdapter
from tbvision.core.config import Settings


class RetrievalService:
    def __init__(
        self,
        settings: Settings,
        vector_db: VectorDBAdapter,
        embedding_service: EmbeddingAdapter,
    ):
        self.settings = settings
        self.vector_db = vector_db
        self.embedding_service = embedding_service
        self.collection_name = settings.knowledge_collection
        self._loaded = True

    async def retrieve(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not query:
            return []

        query_embedding = await self.embedding_service.embed_text(query)
        results = await self.vector_db.search_similar(
            query_embedding=query_embedding,
            top_k=top_k,
            collection_name=self.collection_name,
            filters=filters or {},
        )

        return results or []

    def available(self) -> bool:
        return self._loaded
