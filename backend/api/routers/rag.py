"""Endpoints that expose the RAG helpers."""

from fastapi import APIRouter, Request

from api.schemas import RAGRequest, RAGResponse

router = APIRouter()


@router.post("/rag", response_model=RAGResponse)
def rag_search(request: Request, payload: RAGRequest):
    rag_service = request.app.state.rag_service
    documents = rag_service.retrieve(payload.question, top_k=payload.top_k)
    return RAGResponse(question=payload.question, documents=documents)
