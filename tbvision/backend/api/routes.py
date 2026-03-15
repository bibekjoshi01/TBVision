"""Central router that mounts all sub routers."""

from fastapi import APIRouter

from tbvision.backend.api.routers import health, predict, rag

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(predict.router)
api_router.include_router(rag.router)
