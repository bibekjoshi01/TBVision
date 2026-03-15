"""Endpoints for running model predictions."""

import json
from typing import Any, Optional
from urllib.parse import urljoin

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import ValidationError

from tbvision.api.schemas import PredictionResponse, PredictionMetadata
from tbvision.services.classifier import ClassifierService
from tbvision.services.analyzer import Analyzer
from tbvision.utils.image import content_type_is_image, decode_upload_image
from uuid import uuid4

router = APIRouter()


def _summarize_evidence(evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return "No retrieval evidence was recorded."
    pieces = []
    for entry in evidence[:3]:
        text = entry.get("text", "").strip().replace("\n", " ")
        header = (
            entry.get("metadata", {}).get("title")
            or entry.get("metadata", {}).get("source")
            or entry.get("id")
            or "Evidence"
        )
        if text:
            pieces.append(f"{header}: {text}")
    return "\n".join(pieces) or "Evidence was retrieved but could not be summarized."


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    image: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
):
    if not image.content_type or not content_type_is_image(image.content_type):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    classifier_service: ClassifierService = request.app.state.classifier_service
    decoded = await decode_upload_image(image)

    # Parsing the metadata like symptoms
    parsed_metadata: PredictionMetadata | None = None

    if metadata:
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400, detail=f"metadata must be valid JSON: {exc}"
            )
        if not isinstance(metadata_dict, dict):
            raise HTTPException(
                status_code=400, detail="metadata must be a JSON object"
            )
        try:
            parsed_metadata = PredictionMetadata.model_validate(metadata_dict)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors())

    # Generating the image analysis using baseline model and online llm
    analyzer = Analyzer(
        image=decoded,
        settings=request.app.state.config,
        classifier_service=classifier_service,
        retrieval_service=request.app.state.retrieval_service,
        generation_service=request.app.state.generation_service,
        metadata=parsed_metadata or {},
    )
    pred_data = await analyzer.analyze()
    gradcam_url = pred_data.get("gradcam_image")
    if gradcam_url and not gradcam_url.startswith("http"):
        pred_data["gradcam_image"] = urljoin(
            str(request.base_url), gradcam_url.lstrip("/")
        )

    evidence_summary = _summarize_evidence(pred_data.get("evidence", []))
    summary_text = (pred_data.get("explanation") or "").strip()

    report_id = uuid4().hex
    request.app.state.report_store.save_report(
        report_id=report_id,
        metadata=parsed_metadata.model_dump() if parsed_metadata else {},
        analysis=pred_data,
        summary=summary_text or None,
        evidence_summary=evidence_summary,
        uncertainty_level=pred_data.get("uncertainty_level"),
        uncertainty_std=pred_data.get("uncertainty_std"),
        gradcam_region=pred_data.get("gradcam_region"),
    )

    return PredictionResponse(
        label_map=classifier_service.label_map,
        metadata=parsed_metadata,
        report_id=report_id,
        **pred_data,
    )
