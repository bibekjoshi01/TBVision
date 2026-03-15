from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any


class VectorDBAdapter(ABC):
    @abstractmethod
    async def store_embedding(
        self,
        id: str,
        embedding: list[float],
        metadata: dict[str, Any],
        collection_name: str,
    ):
        pass

    @abstractmethod
    async def search_similar(
        self,
        query_embedding: list[float],
        top_k: int,
        collection_name: str,
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    async def delete_by_document_id(self, document_id: str, collection_name: str):
        pass

    @abstractmethod
    async def ensure_collection(self, collection_name: str) -> None:
        pass

    @abstractmethod
    async def upsert_points(self, points: Sequence[Any], collection_name: str) -> None:
        pass
