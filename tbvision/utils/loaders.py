import asyncio
import tempfile
from pathlib import Path

from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader  # type: ignore[import]


def _read_pages_from_path(path: Path) -> list[str]:
    loader = PyPDFLoader(str(path))
    documents = loader.load()
    return [doc.page_content for doc in documents]


async def load_pdf(file: UploadFile | str | Path) -> list[str]:
    """Async wrapper around PyPDFLoader for UploadFile or filesystem paths."""

    def _load_bytes(file_bytes: bytes) -> list[str]:
        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file.flush()

            try:
                loader = PyPDFLoader(tmp_file.name)
                documents = loader.load()
            except Exception as exc:
                raise ValueError(f"PDF parsing failed: {exc}") from exc

        return [doc.page_content for doc in documents]

    loop = asyncio.get_running_loop()

    if isinstance(file, UploadFile):
        if file.content_type != "application/pdf":
            raise ValueError("Only PDF files are supported")
        file_bytes = await file.read()
        return await loop.run_in_executor(None, _load_bytes, file_bytes)

    path = Path(file)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    return await loop.run_in_executor(None, _read_pages_from_path, path)
