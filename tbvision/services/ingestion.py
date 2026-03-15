import logging
import uuid
from typing import Any

from fastapi import UploadFile
from qdrant_client.http.models import PointStruct

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
        chunk_method: str = "recursive",
    ):
        self.vector_db = vector_db
        self.embedding_service = embedding_service
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_method = chunk_method

    async def ingest_document(
        self,
        document_id: str,
        file: UploadFile,
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
        except Exception as e:
            logger.exception(f"Error loading the PDF: {document_id}")
            raise RuntimeError("Error loading the PDF") from e

        # Split text into chunks
        chunks = await chunk_text(
            full_text,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            method=self.chunk_method,
            return_metadata=True,
            document_id=document_id,
        )

        # Generate embeddings in parallel and store in batches
        EMBED_BATCH_SIZE = 32  # tune for CPU / RAM
        UPSERT_BATCH_SIZE = 64  # tune for network
        points_buffer: list[PointStruct] = []

        await self.vector_db.ensure_collection(collection_name)

        for i in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch_chunks = chunks[i : i + EMBED_BATCH_SIZE]
            texts = [c["text"] for c in batch_chunks]

            embeddings = await self.embedding_service.embed_texts(texts)

            for chunk, embedding in zip(batch_chunks, embeddings, strict=True):
                chunk_id = f"{uuid.uuid4()}"
                payload = {
                    "document_id": document_id,
                    "chunk_index": chunk["chunk_index"],
                    "content": chunk["text"],
                    **(metadata or {}),
                }

                points_buffer.append(
                    PointStruct(
                        id=chunk_id,
                        vector=embedding,
                        payload=payload,
                    )
                )

            if len(points_buffer) >= UPSERT_BATCH_SIZE:
                await self.vector_db.upsert_points(points_buffer, collection_name)
                points_buffer.clear()

        if points_buffer:
            await self.vector_db.upsert_points(points_buffer, collection_name)

    async def delete_existing_syllabus(self, document_id: str, collection_name: str):
        """
        Remove all vectors associated with the given document_id from vector DB.
        """
        await self.vector_db.delete_by_document_id(document_id, collection_name)
