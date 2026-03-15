"""Environment-backed configuration for the TBVision backend."""
from pathlib import Path
from typing import List

from pydantic import BaseSettings, Field, validator


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = ROOT_DIR / "weights" / "ensemble-densenet121_best.pth"
DEFAULT_RAG_DOCS = Path(__file__).resolve().parents[1] / "rag" / "knowledge.json"


class AppConfig(BaseSettings):
    checkpoint_path: Path = Field(default_factory=lambda: DEFAULT_CHECKPOINT)
    image_size: int = Field(224, ge=64, le=1024)
    mode: str = Field("ensemble")
    backbones: List[str] = Field(default_factory=lambda: ["densenet121", "efficientnet_b3", "resnet50"])
    dropout: float = Field(0.3, ge=0.0, le=0.9)
    use_mc_dropout: bool = Field(False)
    allowed_origins: List[str] = Field(default_factory=lambda: ["*"])
    rag_docs_path: Path = Field(default_factory=lambda: DEFAULT_RAG_DOCS)
    rag_top_k: int = Field(3, ge=1, le=10)
    enable_rag: bool = Field(True)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @validator("backbones", pre=True)
    def _split_backbones(cls, value):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @validator("checkpoint_path", "rag_docs_path", pre=True)
    def _resolve_paths(cls, value):
        return Path(value).expanduser()
