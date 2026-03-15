from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT_DIR / "weights" / "xraytb_net.pth"
DEFAULT_KNOWLEDGE_DIR = ROOT_DIR / "tbvision" / "knowledge"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    checkpoint_path: Path = Field(default_factory=lambda: DEFAULT_CHECKPOINT)
    image_size: int = Field(224, ge=64, le=1024)
    mode: str = Field("ensemble")

    backbones: List[str] = Field(
        default_factory=lambda: ["densenet121", "efficientnet_b3", "resnet50"]
    )
    dropout: float = Field(0.3, ge=0.0, le=0.9)
    use_mc_dropout: bool = Field(False)
    media_root: Path = Field(default_factory=lambda: ROOT_DIR / "tbvision" / "static")
    media_url: str = Field("/media")

    app_env: str = "development"
    allowed_origins: List[str] = Field(default_factory=lambda: ["*"])

    # Vector DB
    qdrant_url: str | None = None
    rag_docs_path: Path = Field(default_factory=lambda: DEFAULT_KNOWLEDGE_DIR)
    knowledge_collection: str = Field("tbvision_knowledge")
    context_db_path: Path = Field(
        default_factory=lambda: ROOT_DIR / "tbvision" / "reports.db"
    )

    # Embeddings
    embedding_provider: str = "sentence_transformers"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # LLM
    gemini_api_key: SecretStr | None = None
    mistral_api_key: SecretStr | None = None

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval
    top_k: int = 5

    @field_validator("backbones", mode="before")
    @classmethod
    def split_backbones(cls, value):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator(
        "checkpoint_path",
        "rag_docs_path",
        "media_root",
        "context_db_path",
        mode="before",
    )
    @classmethod
    def resolve_paths(cls, value):
        return Path(value).expanduser()

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v, info):
        if v >= info.data["chunk_size"]:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
