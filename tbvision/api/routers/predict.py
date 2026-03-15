"""Endpoints for running model predictions."""

import json
from typing import Optional
from urllib.parse import urljoin

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from tbvision.api.schemas import PredictionResponse, PredictionMetadata
from tbvision.services.classifier import ClassifierService
from tbvision.services.analyzer import Analyzer
from tbvision.utils.image import content_type_is_image, decode_upload_image

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    image: UploadFile = File(...),
    metadata: Optional[PredictionMetadata] = Form(None),
):
    if not image.content_type or not content_type_is_image(image.content_type):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    classifier_service: ClassifierService = request.app.state.classifier_service
    decoded = await decode_upload_image(image)

    # Parsing the metadata like symptoms
    parsed_metadata = None
    if metadata:
        try:
            parsed_metadata = json.loads(metadata)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400, detail=f"metadata must be valid JSON: {exc}"
            )
        if not isinstance(parsed_metadata, dict):
            raise HTTPException(
                status_code=400, detail="metadata must be a JSON object"
            )

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

    return PredictionResponse(
        label_map=classifier_service.label_map,
        metadata=parsed_metadata,
        **pred_data,
    )
