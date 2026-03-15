from functools import lru_cache

from tbvision.adapters.embeddings.base import EmbeddingAdapter
from tbvision.adapters.embeddings.sentence_transformers_adapter import (
    SentenceTransformersAdapter,
)
from tbvision.adapters.vector_db.base import VectorDBAdapter
from tbvision.adapters.vector_db.qdrant_adapter import QdrantAdapter
from tbvision.core.config import get_settings

settings = get_settings()


@lru_cache
def get_vector_db() -> VectorDBAdapter:
    return QdrantAdapter(
        url=settings.qdrant_url or "http://localhost:6333",
        vector_size=settings.embedding_dimension,
    )


@lru_cache
def get_embedding_service() -> EmbeddingAdapter:
    if settings.embedding_provider == "sentence_transformers":
        return SentenceTransformersAdapter(model_name=settings.embedding_model)

    else:
        raise ValueError(
            f"Unsupported embedding provider: {settings.embedding_provider}"
        )
