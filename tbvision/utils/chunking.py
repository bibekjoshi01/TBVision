import asyncio

from langchain_classic.text_splitter import RecursiveCharacterTextSplitter


async def chunk_text(
    content: str,
    chunk_size: int,
    chunk_overlap: int,
    return_metadata: bool = False,
    document_id: str | None = None,
) -> list:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    def _sync_split() -> list:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return splitter.split_text(content)

    loop = asyncio.get_running_loop()
    chunks = await loop.run_in_executor(None, _sync_split)

    filtered = [chunk for chunk in chunks if chunk.strip()]
    if not return_metadata:
        return filtered

    results = []
    
    for index, chunk in enumerate(filtered):
        chunk_obj = {
            "chunk_index": index,
            "text": chunk,
        }
        if document_id:
            chunk_obj["document_id"] = document_id
        results.append(chunk_obj)
    
    return results
