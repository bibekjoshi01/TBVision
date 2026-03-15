"""Health endpoints for the API."""

from fastapi import APIRouter, Request

from tbvision.api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    classifier_service = request.app.state.classifier_service
    config = request.app.state.config

    return HealthResponse(
        status="ok" if classifier_service.ready else "loading",
        checkpoint=str(config.checkpoint_path),
        device=classifier_service.device,
        rag_enabled=request.app.state.rag_service.available(),
    )
