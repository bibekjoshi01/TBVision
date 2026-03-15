from functools import lru_cache

from tbvision.adapters.embeddings.base import EmbeddingAdapter
from tbvision.adapters.embeddings.sentence_transformers_adapter import (
    SentenceTransformersAdapter,
)
from tbvision.adapters.llm.base import LLMAdapter
from tbvision.adapters.llm.gemini_adapter import GeminiAdapter
from tbvision.adapters.llm.local_adapter import LocalAdapter
from tbvision.adapters.llm.mistral_adapter import MistralAdapter
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


@lru_cache
def get_llm_service() -> LLMAdapter:
    if settings.llm_provider == "gemini":
        api_key = settings.gemini_api_key
        if not api_key:
            raise ValueError("Gemini API key is required for remote LLMs.")
        return GeminiAdapter(api_key=api_key.get_secret_value())

    elif settings.llm_provider == "local":
        model_name = settings.local_llm_model

        if not model_name:
            raise ValueError("Local LLM model name must be configured.")
        return LocalAdapter(model_name=model_name)

    elif settings.llm_provider == "mistral":
        api_key = settings.mistral_api_key

        if not api_key:
            raise ValueError("Mistral API key is required for remote LLMs.")
        return MistralAdapter(api_key=api_key.get_secret_value())

    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
