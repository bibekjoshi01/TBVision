"""Pydantic models for API requests and responses."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PatientInfo(BaseModel):
    age: Optional[int] = None
    sex: Optional[str] = None
    region: Optional[str] = None


class Symptoms(BaseModel):
    cough: Optional[bool] = None
    cough_duration_days: Optional[int] = None
    fever: Optional[bool] = None
    night_sweats: Optional[bool] = None
    weight_loss: Optional[bool] = None
    chest_pain: Optional[bool] = None
    shortness_of_breath: Optional[bool] = None
    fatigue: Optional[bool] = None


class RiskFactors(BaseModel):
    smoker: Optional[bool] = None
    diabetes: Optional[bool] = None
    hiv_positive: Optional[bool] = None
    close_contact_tb_patient: Optional[bool] = None
    immunocompromised: Optional[bool] = None


class MedicalHistory(BaseModel):
    previous_tb: Optional[bool] = None
    chronic_lung_disease: Optional[bool] = None
    recent_pneumonia: Optional[bool] = None


class ScreeningContext(BaseModel):
    screening_type: Optional[str] = None
    location: Optional[str] = None


class PredictionMetadata(BaseModel):
    patient_info: Optional[PatientInfo] = None
    symptoms: Optional[Symptoms] = None
    risk_factors: Optional[RiskFactors] = None
    medical_history: Optional[MedicalHistory] = None
    screening_context: Optional[ScreeningContext] = None


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
    metadata: Optional[PredictionMetadata] = None
    report_id: Optional[str] = None
    gradcam_region: Optional[str] = None
    gradcam_image: Optional[str] = None
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
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


class FollowUpHistoryEntry(BaseModel):
    question: str
    answer: str
    created_at: str


class FollowUpRequest(BaseModel):
    report_id: str = Field(..., min_length=8)
    question: str = Field(..., min_length=5)


class FollowUpResponse(BaseModel):
    report_id: str
    question: str
    answer: str
    history: List[FollowUpHistoryEntry]


class FollowUpHistoryResponse(BaseModel):
    report_id: str
    history: List[FollowUpHistoryEntry]
