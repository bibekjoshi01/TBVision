"""Endpoints that expose the retrieval helpers."""

from fastapi import APIRouter, Request

from tbvision.api.schemas import RAGRequest, RAGResponse
from tbvision.services.retrieval import RetrievalService

router = APIRouter()


@router.post("/rag", response_model=RAGResponse)
async def rag_search(request: Request, payload: RAGRequest):
    retrieval_service: RetrievalService = request.app.state.retrieval_service
    top_k = payload.top_k or request.app.state.config.top_k
    documents = await retrieval_service.retrieve(
        payload.question, top_k=top_k, filters={}
    )
    normalized = [
        {
            "id": doc.get("id") or str(idx),
            "text": doc.get("content") or "",
            "metadata": doc.get("metadata", {}),
            "score": doc.get("score", 0.0),
        }
        for idx, doc in enumerate(documents)
    ]
    return RAGResponse(question=payload.question, documents=normalized)
