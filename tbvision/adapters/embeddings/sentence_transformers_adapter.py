import asyncio

from sentence_transformers import SentenceTransformer  # type: ignore[import]

from tbvision.adapters.embeddings.base import EmbeddingAdapter


class SentenceTransformersAdapter(EmbeddingAdapter):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    async def embed_text(self, text: str) -> list[float]:
        embeddings = await self.embed_texts([text])
        return embeddings[0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        result = await asyncio.to_thread(self.model.encode, texts)
        return [list(vec) for vec in result]
