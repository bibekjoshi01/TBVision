"""Application factory for the TBVision FastAPI backend."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import api_router
from .core.config import AppConfig
from .core.logging import configure_logging
from .rag.service import RAGService
from .services.classifier import ClassifierService


def create_app() -> FastAPI:
    config = AppConfig()
    configure_logging()

    app = FastAPI(
        title="TBVision Backend",
        version="1.0.0",
        description="Classification and retrieval API for the TBVision CXR model.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.allowed_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    app.state.config = config
    app.state.classifier_service = ClassifierService(config)
    app.state.rag_service = RAGService(config)

    @app.on_event("startup")
    async def _startup():
        app.state.classifier_service.load()
        app.state.rag_service.load()

    app.include_router(api_router)
    return app
