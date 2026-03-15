"""FastAPI inference service for TBVision."""
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ai_model.inference import ClassificationService


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT_DIR / "weights" / "ensemble-densenet121_best.pth"

IMAGE_SIZE = int(os.getenv("TBVISION_IMAGE_SIZE", "224"))
CHECKPOINT_PATH = Path(os.getenv("TBVISION_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
MODE = os.getenv("TBVISION_MODE", "ensemble")
BACKBONES = [b.strip() for b in os.getenv("TBVISION_BACKBONES", "densenet121,efficientnet_b3,resnet50").split(",") if b.strip()]
DROPOUT = float(os.getenv("TBVISION_DROPOUT", "0.3"))
USE_MC_DROPOUT = os.getenv("TBVISION_USE_MC_DROPOUT", "false").lower() in ("1", "true", "yes")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="TBVision Inference API", version="1.0.0", description="Serve CXR predictions.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

classification_service: Optional[ClassificationService] = None


class PredictionResponse(BaseModel):
    prediction: str
    probabilities: Dict[str, float]
    raw_logits: List[float]
    label_map: List[str]
    metadata: Optional[Dict[str, Any]] = None
    model_config: Dict[str, Any] = Field(..., description="Snapshot of the configuration used for the request.")


class HealthResponse(BaseModel):
    status: str
    checkpoint: str
    device: str


@app.on_event("startup")
def startup_event():
    global classification_service
    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. Set TBVISION_CHECKPOINT to a valid file."
        )
    classification_service = ClassificationService(
        checkpoint_path=CHECKPOINT_PATH,
        image_size=IMAGE_SIZE,
        mode=MODE,
        backbones=BACKBONES,
        dropout=DROPOUT,
        use_mc_dropout=USE_MC_DROPOUT,
    )
    logging.info("Loaded checkpoint %s (%s)", CHECKPOINT_PATH, classification_service.device)


def parse_metadata(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"metadata must be valid JSON: {exc}")
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="metadata must be a JSON object")
    return parsed


async def _decode_image(upload: UploadFile) -> np.ndarray:
    contents = await upload.read()
    array = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise HTTPException(status_code=400, detail="Unable to decode the uploaded image")
    return image


async def _run_prediction(image: np.ndarray) -> Dict[str, Any]:
    if classification_service is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return await asyncio.to_thread(classification_service.predict, image)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if classification_service is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return HealthResponse(
        status="ok",
        checkpoint=str(CHECKPOINT_PATH),
        device=classification_service.device,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")
    image_array = await _decode_image(image)
    raw_response = await _run_prediction(image_array)
    metadata_payload = parse_metadata(metadata)
    return PredictionResponse(
        prediction=raw_response["prediction"],
        probabilities=raw_response["probabilities"],
        raw_logits=raw_response["raw_logits"],
        label_map=classification_service.label_map if classification_service else [],
        metadata=metadata_payload,
        model_config={
            "checkpoint": str(CHECKPOINT_PATH),
            "image_size": IMAGE_SIZE,
            "mode": MODE,
            "backbones": BACKBONES,
            "dropout": DROPOUT,
            "use_mc_dropout": USE_MC_DROPOUT,
        },
    )
