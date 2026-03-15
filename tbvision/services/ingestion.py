import logging
import uuid
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from tbvision.adapters.embeddings.base import EmbeddingAdapter
from tbvision.adapters.vector_db.base import VectorDBAdapter
from tbvision.utils.chunking import chunk_text
from tbvision.utils.loaders import load_pdf

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(
        self,
        vector_db: VectorDBAdapter,
        embedding_service: EmbeddingAdapter,
        chunk_size: int,
        chunk_overlap: int,
    ):
        self.vector_db = vector_db
        self.embedding_service = embedding_service
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def _process_text(
        self,
        document_id: str,
        text: str,
        metadata: dict[str, Any] | None,
        collection_name: str,
    ):
        if not text.strip():
            logger.warning("Document %s contains no textual content.", document_id)
            return

        chunks = await chunk_text(
            text,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            return_metadata=True,
            document_id=document_id,
        )

        if not chunks:
            return

        await self.vector_db.ensure_collection(collection_name)

        texts = [chunk["text"] for chunk in chunks]
        embeddings = await self.embedding_service.embed_texts(texts)

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            chunk_id = f"{uuid.uuid4()}"
            payload = {
                "document_id": document_id,
                "chunk_index": chunk["chunk_index"],
                "content": chunk["text"],
                **(metadata or {}),
            }
            await self.vector_db.store_embedding(
                chunk_id,
                list(embedding),
                payload,
                collection_name,
            )

    async def ingest_document(
        self,
        document_id: str,
        file: UploadFile | Path | str,
        metadata: dict[str, Any] | None = None,
        collection_name: str = "default_documents",
    ):
        """
        Ingest a PDF document into vector DB:
        - Delete old vectors for this document
        - Extract PDF text
        - Chunk text
        - Generate embeddings
        - Store embeddings in vector DB
        """

        # Delete old embeddings
        await self.delete_existing_syllabus(document_id, collection_name)

        # Load PDF pages
        try:
            pages = await load_pdf(file)
            full_text = "\n".join(pages)
        except Exception as exc:
            logger.exception("Error loading the PDF: %s", document_id)
            raise RuntimeError("Error loading the PDF") from exc

        await self._process_text(document_id, full_text, metadata, collection_name)

    async def delete_existing_syllabus(self, document_id: str, collection_name: str):
        """
        Remove all vectors associated with the given document_id from vector DB.
        """
        await self.vector_db.delete_by_document_id(document_id, collection_name)
