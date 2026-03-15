"""Script that indexes the `knowledge/` PDFs into the retrieval vector store."""

from __future__ import annotations

import asyncio
import logging

from tbvision.core.config import get_settings
from tbvision.core.dependencies import get_embedding_service, get_vector_db
from tbvision.services.ingestion import IngestionService

logger = logging.getLogger(__name__)


async def main() -> None:
    settings = get_settings()
    vector_db = get_vector_db()
    embedding_service = get_embedding_service()
    ingestion_service = IngestionService(
        vector_db=vector_db,
        embedding_service=embedding_service,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    knowledge_dir = settings.rag_docs_path
    pdf_paths = sorted(knowledge_dir.glob("*.pdf"))
    if not pdf_paths:
        logger.warning("No knowledge PDFs found in %s", knowledge_dir)
        return

    logger.info("Indexing %d knowledge PDFs from %s", len(pdf_paths), knowledge_dir)
    for pdf in pdf_paths:
        logger.info("Ingesting %s", pdf.name)
        await ingestion_service.ingest_document(
            document_id=pdf.stem,
            file=pdf,
            collection_name=settings.knowledge_collection,
        )

    logger.info("Knowledge base indexing complete (%d documents).", len(pdf_paths))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    asyncio.run(main())
