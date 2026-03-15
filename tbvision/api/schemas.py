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
    prediction_label: str
    probability: float
    probabilities: Dict[str, float]
    raw_logits: List[float]
    label_map: List[str]
    uncertainty_std: float
    uncertainty_level: str
    metadata: Optional[Dict[str, Any]] = None
    gradcam_region: Optional[str] = None
    gradcam_image: Optional[str] = None
    evidence: List[Dict[str, Any]]
    explanation: str


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
