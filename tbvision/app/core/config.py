from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[4]
DEFAULT_CHECKPOINT = ROOT_DIR / "weights" / "ensemble-densenet121-efficientnet_b3-resnet50_best.pth"
DEFAULT_RAG_DOCS = ROOT_DIR / "tbvision" / "backend" / "knowledge" / "knowledge.json"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
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

    app_env: str = "development"
    allowed_origins: List[str] = Field(default_factory=lambda: ["*"])

    rag_docs_path: Path = Field(default_factory=lambda: DEFAULT_RAG_DOCS)
    rag_top_k: int = Field(3, ge=1, le=10)
    enable_rag: bool = Field(True)

    @field_validator("backbones", mode="before")
    @classmethod
    def split_backbones(cls, value):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("checkpoint_path", "rag_docs_path", mode="before")
    @classmethod
    def resolve_paths(cls, value):
        return Path(value).expanduser()


@lru_cache
def get_settings() -> Settings:
    return Settings()
