"""Pydantic models for API requests and responses."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    checkpoint: str
    device: str
    rag_enabled: bool


class PredictionResponse(BaseModel):
    prediction: str
    probabilities: Dict[str, float]
    raw_logits: List[float]
    label_map: List[str]
    metadata: Optional[Dict[str, Any]] = None
    config_snapshot: Dict[str, Any]


class RAGDocument(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any]
    score: float


class RAGRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: Optional[int] = Field(None, ge=1, le=10)


class RAGResponse(BaseModel):
    question: str
    documents: List[RAGDocument]
