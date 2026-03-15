"""Wrapper around the AI model for dependency injection."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from tbvision.xraytb_net.inference import ClassificationService
from tbvision.app.core.config import Settings


logger = logging.getLogger(__name__)


class ClassifierService:
    def __init__(self, config: Settings):
        self.config = config
        self._service: Optional[ClassificationService] = None

    def load(self) -> None:
        if not Path(self.config.checkpoint_path).exists():
            raise RuntimeError(
                f"Checkpoint not found at {self.config.checkpoint_path}."
            )
        self._service = ClassificationService(
            checkpoint_path=self.config.checkpoint_path,
            image_size=self.config.image_size,
            mode=self.config.mode,
            backbones=self.config.backbones,
            dropout=self.config.dropout,
            use_mc_dropout=self.config.use_mc_dropout,
        )
        logger.info(
            "Loaded %s on %s",
            self.config.checkpoint_path,
            self.device,
        )

    @property
    def ready(self) -> bool:
        return self._service is not None

    @property
    def device(self) -> str:
        if not self._service:
            return "cpu"
        return self._service.device

    @property
    def label_map(self) -> List[str]:
        return self._service.label_map if self._service else []

    def predict(self, image: np.ndarray) -> Dict[str, Optional[float]]:
        if not self._service:
            raise RuntimeError("ClassifierService has not been loaded yet.")
        return self._service.predict(image)
