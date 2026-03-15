"""Serving utilities for multi-class CXR inference and evaluation."""

import cv2
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from ..data_preparation.transforms import get_val_transforms
from ..models.classifier import TBClassifier
from ..models.ensemble import TBEnsemble


DEFAULT_LABELS = ["Normal", "OtherDisease", "TB"]


class ClassificationService:
    """Lightweight inference wrapper for the 3-class CXR classifier."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[str] = None,
        image_size: int = 224,
        mode: str = "ensemble",
        backbones: Optional[List[str]] = None,
        dropout: float = 0.3,
        use_mc_dropout: bool = False,
        label_map: Optional[List[str]] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.transforms = get_val_transforms(image_size)
        self.label_map = label_map or DEFAULT_LABELS
        self.model = self._build_model(
            mode=mode,
            backbones=backbones,
            dropout=dropout,
            use_mc_dropout=use_mc_dropout,
        ).to(self.device)
        self.load_checkpoint(checkpoint_path)
        self.model.eval()

    def _build_model(
        self,
        mode: str,
        backbones: Optional[List[str]],
        dropout: float,
        use_mc_dropout: bool,
    ) -> torch.nn.Module:
        if mode == "ensemble":
            return TBEnsemble(
                backbones=backbones,
                dropout_rate=dropout,
                use_mc_dropout=use_mc_dropout,
            )
        backbone = backbones[0] if backbones else "densenet121"
        return TBClassifier(backbone=backbone, dropout=dropout)

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    def _read_image(self, image: Union[str, np.ndarray]) -> np.ndarray:
        if isinstance(image, str):
            image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Unable to read image at {image}")
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def preprocess(self, image: Union[str, np.ndarray]) -> torch.Tensor:
        image = self._read_image(image)
        transformed = self.transforms(image=image)["image"]
        return transformed

    def predict(
        self, image: Union[str, np.ndarray], probability_threshold: float = 0.5
    ) -> Dict[str, Union[str, float, Dict[str, float]]]:
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        image_tensor = tensor.clone()

        # Ensure single channel
        if image_tensor.shape[1] == 3:
            image_tensor = image_tensor.mean(dim=1, keepdim=True)
        elif image_tensor.shape[1] != 1:
            image_tensor = image_tensor[:, :1, :, :]

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        target_label = "TB"
        if target_label in self.label_map:
            target_index = self.label_map.index(target_label)
        else:
            target_index = int(np.argmax(probs))

        mean_prob = float(probs[target_index])
        std_prob = 0.0
        if getattr(self.model, "use_mc_dropout", False):
            mean_preds, std_preds = self.model.predict_with_uncertainty(
                image_tensor, n_samples=20
            )
            mean_preds = mean_preds.detach().cpu()
            std_preds = std_preds.detach().cpu()
            mean_prob = float(mean_preds[0, target_index])
            std_prob = float(std_preds[0, target_index])

        if std_prob < 0.12:
            uncertainty = "Low"
        elif std_prob < 0.20:
            uncertainty = "Medium"
        else:
            uncertainty = "High"

        prediction_label = (
            "Possible Tuberculosis"
            if mean_prob >= probability_threshold
            else "Likely Normal"
        )

        result = {
            "raw_logits": logits.cpu().numpy().tolist()[0],
            "probabilities": {
                cls: float(probs[idx]) for idx, cls in enumerate(self.label_map)
            },
            "prediction": self.label_map[int(np.argmax(probs))],
            # ----------------------------------
            "probability": mean_prob,
            "prediction_label": prediction_label,
            "uncertainty_std": std_prob,
            "uncertainty_level": uncertainty,
            "image_tensor": image_tensor,
        }

        return result
