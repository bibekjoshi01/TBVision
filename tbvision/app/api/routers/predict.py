"""Endpoints for running model predictions."""

import json
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from tbvision.app.api.schemas import PredictionResponse
from tbvision.app.services.classifier import ClassifierService
from tbvision.app.services.image import content_type_is_image, decode_upload_image

router = APIRouter()


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
    raw = classifier_service.predict(decoded)

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

    return PredictionResponse(
        prediction=raw["prediction"],
        probabilities=raw["probabilities"],
        raw_logits=raw["raw_logits"],
        label_map=classifier_service.label_map,
        metadata=parsed_metadata,
        config_snapshot={
            "checkpoint": str(request.app.state.config.checkpoint_path),
            "image_size": request.app.state.config.image_size,
            "mode": request.app.state.config.mode,
            "backbones": request.app.state.config.backbones,
            "dropout": request.app.state.config.dropout,
            "use_mc_dropout": request.app.state.config.use_mc_dropout,
        },
    )
