"""Light retrieval-augmented generation utilities."""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from tbvision.core.config import Settings


logger = logging.getLogger(__name__)


def _embed_text(text: str, dimension: int = 256) -> np.ndarray:
    vector = np.zeros(dimension, dtype=float)
    tokens = text.lower().split()
    for token in tokens:
        idx = abs(hash(token)) % dimension
        vector[idx] += 1
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    if np.all(vec_a == 0) or np.all(vec_b == 0):
        return 0.0
    return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))


@dataclass
class Document:
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(256))


class KnowledgeBase:
    def __init__(self, documents: List[Document]):
        self.documents = documents

    @classmethod
    def load(cls, path: Path) -> "KnowledgeBase":
        if not path.exists():
            raise FileNotFoundError(f"Knowledge document {path} does not exist")
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        documents = []
        for item in raw:
            text = item.get("text", "")
            doc_id = item.get("id", text[:32])
            metadata = item.get("metadata", {})
            documents.append(
                Document(
                    id=doc_id,
                    text=text,
                    metadata=metadata,
                    embedding=_embed_text(text),
                )
            )
        return cls(documents)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_vec = _embed_text(query)
        scored = []
        for doc in self.documents:
            score = _cosine_similarity(query_vec, doc.embedding)
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "id": doc.id,
                "text": doc.text,
                "metadata": doc.metadata,
                "score": float(score),
            }
            for score, doc in scored[:top_k]
        ]


class RAGService:
    def __init__(self, config: Settings):
        self._config = config
        self._knowledge: Optional[KnowledgeBase] = None

    def load(self) -> None:
        if not self._config.enable_rag:
            logger.info("RAG service disabled via configuration")
            self._knowledge = None
            return
        try:
            self._knowledge = KnowledgeBase.load(self._config.rag_docs_path)
            logger.info("Loaded %d RAG documents", len(self._knowledge.documents))
        except FileNotFoundError as exc:
            logger.warning("RAG documents not loaded: %s", exc)
            self._knowledge = None

    def available(self) -> bool:
        return self._knowledge is not None

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self._knowledge:
            return []
        top_k = top_k or self._config.rag_top_k
        return self._knowledge.retrieve(query, top_k=top_k)
