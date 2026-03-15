from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from tbvision.api.routes import api_router
from tbvision.core.config import get_settings
from tbvision.core.logging import configure_logging
from tbvision.services.rag import RAGService
from tbvision.services.classifier import ClassifierService

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Configure logging first
    configure_logging()

    # Initialize services
    classifier_service = ClassifierService(settings)
    rag_service = RAGService(settings)

    # Load models / indexes
    classifier_service.load()
    rag_service.load()

    # Store in app state
    app.state.config = settings
    app.state.classifier_service = classifier_service
    app.state.rag_service = rag_service

    yield  # App runs here


app = FastAPI(
    title="TBVision Backend",
    description="Classification and retrieval API for the TBVision CXR model.",
    version="1.0.0",
    docs_url="/docs" if settings.app_env != "production" else None,
    redoc_url="/redoc" if settings.app_env != "production" else None,
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


app.include_router(api_router, prefix="/api")


@app.get("/", tags=["meta"], status_code=200)
async def root() -> dict[str, str]:
    return {
        "service": "X Ray TB Backend Service",
        "status": "ok",
    }
