import logging
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader  # type: ignore[import]

from tbvision.adapters.embeddings.base import EmbeddingAdapter
from tbvision.adapters.vector_db.base import VectorDBAdapter
from tbvision.core.config import Settings
from tbvision.utils.chunking import chunk_text

logger = logging.getLogger(__name__)


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
        self.documents_path = settings.rag_docs_path
        self._loaded = False

    async def load(self) -> None:
        if self._loaded:
            return

        if not self.documents_path.exists():
            logger.warning(
                "Knowledge directory %s does not exist. Retrieval data unavailable.",
                self.documents_path,
            )
            self._loaded = True
            return

        pdf_files = list(self.documents_path.glob("*.pdf"))
        if not pdf_files:
            logger.warning("No knowledge PDFs found in %s", self.documents_path)
            self._loaded = True
            return

        for pdf_path in pdf_files:
            document_id = pdf_path.stem
            try:
                loader = PyPDFLoader(str(pdf_path))
                docs = loader.load()
            except Exception as exc:
                logger.error("Failed to load %s: %s", pdf_path, exc, exc_info=True)
                continue

            text = "\n".join(doc.page_content for doc in docs if doc.page_content)
            if not text:
                continue

            await self.vector_db.delete_by_document_id(
                document_id, self.collection_name
            )

            chunks = await chunk_text(
                text,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
                return_metadata=True,
                document_id=document_id,
            )

            if not chunks:
                continue

            texts = [chunk["text"] for chunk in chunks]
            embeddings = await self.embedding_service.embed_texts(texts)

            for chunk, embedding in zip(chunks, embeddings, strict=True):
                chunk_index = chunk["chunk_index"]
                metadata = {
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "content": chunk["text"],
                    "source": pdf_path.name,
                }
                await self.vector_db.store_embedding(
                    f"{document_id}-{chunk_index}",
                    list(embedding),
                    metadata,
                    self.collection_name,
                )

            logger.info(
                "Indexed %d chunks from %s",
                len(chunks),
                pdf_path.name,
            )

        self._loaded = True

    async def retrieve(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not self._loaded:
            logger.warning(
                "Retrieval requested before knowledge load completed; returning empty list."
            )
            return []

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
