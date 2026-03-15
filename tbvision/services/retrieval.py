from typing import Any

from tbvision.adapters.embeddings.base import EmbeddingAdapter
from tbvision.adapters.vector_db.base import VectorDBAdapter


class RetrievalService:
    def __init__(
        self,
        vector_db: VectorDBAdapter,
        embedding_service: EmbeddingAdapter,
        collection_name: str,
    ):
        self.vector_db = vector_db
        self.embedding_service = embedding_service
        self.collection_name = collection_name

    async def retrieve(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any],
    ) -> list[dict]:
        """
        Performs syllabus-scoped semantic retrieval.
        """

        if not query:
            raise ValueError("Retrieval query cannot be empty")

        query_embedding = await self.embedding_service.embed_text(query)

        results = await self.vector_db.search_similar(
            query_embedding=query_embedding,
            top_k=top_k,
            collection_name=self.collection_name,
            filters=filters,
        )

        if not results:
            return []

        return results
