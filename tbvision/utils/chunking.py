import asyncio

from langchain_classic.text_splitter import (  # type: ignore[import]
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)


async def chunk_text(
    content: str,
    chunk_size: int,
    chunk_overlap: int,
    return_metadata: bool = False,
    method: str = "recursive",
    document_id: str | None = None,
) -> list:
    """
    Async wrapper to split text into chunks.
    """

    def _sync_chunk() -> list:
        splitter: CharacterTextSplitter | RecursiveCharacterTextSplitter
        if method == "recursive":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
            )
        elif method == "character":
            splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator="\n",
                length_function=len,
            )
        else:
            raise ValueError("Invalid method. Choose 'recursive' or 'character'.")

        chunks = splitter.split_text(content)

        if return_metadata:
            return [
                {
                    "chunk_index": i,
                    "text": chunk,
                    **({"syllabus_uuid": document_id} if document_id else {}),
                }
                for i, chunk in enumerate(chunks)
            ]

        return chunks

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_chunk)
