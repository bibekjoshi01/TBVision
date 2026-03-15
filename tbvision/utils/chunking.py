import asyncio


async def chunk_text(
    content: str,
    chunk_size: int,
    chunk_overlap: int,
    return_metadata: bool = False,
    document_id: str | None = None,
) -> list:
    """
    Async chunking helper splits content by character window.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    def _sync_chunk() -> list:
        results = []
        step = chunk_size - chunk_overlap
        start = 0
        index = 0
        while start < len(content):
            end = min(len(content), start + chunk_size)
            chunk = content[start:end]
            if not chunk.strip():
                start += step
                continue
            if return_metadata:
                chunk_obj = {
                    "chunk_index": index,
                    "text": chunk,
                    **({"syllabus_uuid": document_id} if document_id else {}),
                }
            else:
                chunk_obj = chunk
            results.append(chunk_obj)
            index += 1
            start += step
        return results

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_chunk)
