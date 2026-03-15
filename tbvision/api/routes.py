"""Central router that mounts all sub routers."""

from fastapi import APIRouter

from tbvision.api.routers import followup, health, predict, rag

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(predict.router)
api_router.include_router(rag.router)
api_router.include_router(followup.router)
