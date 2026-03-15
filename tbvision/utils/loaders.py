import asyncio
import tempfile

from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader  # type: ignore[import]


async def load_pdf(file: UploadFile) -> list[str]:
    """
    Async wrapper around PyPDFLoader for FastAPI UploadFile.
    Returns a list of page texts.
    """

    if file.content_type != "application/pdf":
        raise ValueError("Only PDF files are supported")

    def _sync_load(file_bytes: bytes) -> list[str]:
        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file.flush()

            try:
                loader = PyPDFLoader(tmp_file.name)
                documents = loader.load()
            except Exception as e:
                raise ValueError(f"PDF parsing failed: {e}")

        return [doc.page_content for doc in documents]

    file_bytes = await file.read()  # read UploadFile bytes asynchronously
    loop = asyncio.get_running_loop()
    page_texts = await loop.run_in_executor(None, _sync_load, file_bytes)
    return page_texts
