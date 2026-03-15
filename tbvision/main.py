from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from tbvision.api.routes import api_router
from tbvision.core.config import get_settings
from tbvision.core.dependencies import get_embedding_service, get_vector_db
from tbvision.core.logging import configure_logging
from tbvision.services.classifier import ClassifierService
from tbvision.services.generation import GenerationService
from tbvision.services.report_store import ReportStore
from tbvision.services.retrieval import RetrievalService

settings = get_settings()
settings.media_root.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Configure logging first
    configure_logging()

    # Initialize services
    classifier_service = ClassifierService(settings)
    vector_db = get_vector_db()
    embedding_service = get_embedding_service()
    retrieval_service = RetrievalService(settings, vector_db, embedding_service)
    generation_service = GenerationService(settings)
    report_store = ReportStore(settings.context_db_path)

    # Load models
    classifier_service.load()

    # Store in app state
    app.state.config = settings
    app.state.classifier_service = classifier_service
    app.state.retrieval_service = retrieval_service
    app.state.generation_service = generation_service
    app.state.report_store = report_store

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


app.mount(settings.media_url, StaticFiles(directory=settings.media_root), name="media")
app.include_router(api_router, prefix="/api")


@app.get("/", tags=["meta"], status_code=200)
async def root() -> dict[str, str]:
    return {
        "service": "TBVision Backend Service",
        "status": "ok",
    }
