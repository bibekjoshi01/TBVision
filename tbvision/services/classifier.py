"""Wrapper around the AI model for dependency injection."""

import cv2
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from uuid import uuid4

from tbvision.xraytb_net.inference import ClassificationService
from tbvision.core.config import Settings


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

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        if not self._service:
            raise RuntimeError("ClassifierService has not been loaded yet.")
        return self._service.predict(image)

    def analyze_gradcam(self, pred_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute Grad-CAM++ heatmap and highlight the dominant lung region."""

        if not self._service:
            raise RuntimeError("ClassifierService has not been loaded yet.")

        image_tensor = pred_data.get("image_tensor")
        if image_tensor is None or not isinstance(image_tensor, torch.Tensor):
            raise ValueError(
                "`image_tensor` must be present in prediction data for Grad-CAM."
            )

        try:
            from pytorch_grad_cam import GradCAMPlusPlus
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "pytorch-grad-cam is required for Grad-CAM analysis."
            ) from exc

        base_model, target_layer = self._get_gradcam_model_and_layer()
        target_index = self._resolve_target_index(pred_data.get("prediction"))

        cam = GradCAMPlusPlus(
            model=base_model,
            target_layers=[target_layer],
        )

        grayscale_cam = cam(
            input_tensor=image_tensor,
            targets=[ClassifierOutputTarget(target_index)],
        )[0]

        region_info = self._describe_heatmap_regions(grayscale_cam)

        return {
            "dominant_region": region_info["dominant"],
            "description": region_info["description"],
            "heatmap": grayscale_cam,
        }

    def _get_gradcam_model_and_layer(self) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Resolve the base classifier and the convolutional layer to feed Grad-CAM."""

        model = self._service.model
        if hasattr(model, "models") and hasattr(model, "backbones"):
            base = model.models[0]
            backbone_name = model.backbones[0]
        else:
            base = model
            backbone_name = getattr(base, "backbone_name", "densenet121")

        encoder = getattr(base, "encoder", base)
        target_layer = self._select_conv_layer(encoder, backbone_name)
        return base, target_layer

    def _select_conv_layer(
        self, encoder: torch.nn.Module, backbone_name: str
    ) -> torch.nn.Module:
        if hasattr(encoder, "features") and hasattr(encoder.features, "denseblock4"):
            return encoder.features.denseblock4
        if hasattr(encoder, "layer4"):
            return encoder.layer4
        if hasattr(encoder, "blocks") and len(getattr(encoder, "blocks")) > 0:
            return encoder.blocks[-1]
        if hasattr(encoder, "conv_head"):
            return encoder.conv_head
        return encoder

    def _resolve_target_index(self, prediction: Optional[str]) -> int:
        if prediction is None:
            return 0
        try:
            return self.label_map.index(prediction)
        except ValueError:
            return 0

    def _describe_heatmap_regions(self, heatmap: np.ndarray) -> Dict[str, str]:
        h = heatmap.shape[0]
        upper = np.mean(heatmap[: h // 3])
        middle = np.mean(heatmap[h // 3 : 2 * h // 3])
        lower = np.mean(heatmap[2 * h // 3 :])

        regions = {"upper": upper, "middle": middle, "lower": lower}
        dominant = max(regions, key=regions.get)

        if dominant == "upper":
            description = "Upper lung zones (post-primary TB patterns)"
        elif dominant == "lower":
            description = "Lower lung zones"
        else:
            description = "Diffuse distribution across lung fields"

        return {"dominant": dominant, "description": description}

    def create_gradcam_overlay(self, image: np.ndarray, gradcam_heatmap: np.ndarray):
        """Blend a Grad-CAM heatmap with the original X-ray and write it to disk for serving."""

        if image is None or gradcam_heatmap is None:
            return None

        original = image
        if original.dtype != np.uint8:
            original = np.clip(original, 0, 255).astype(np.uint8)

        if original.ndim == 2:
            original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        elif original.ndim == 3 and original.shape[2] == 1:
            original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        elif original.ndim == 3 and original.shape[2] == 4:
            original = original[:, :, :3]

        h, w = original.shape[:2]

        resized_heatmap = cv2.resize(gradcam_heatmap, (w, h))
        scaled_heatmap = np.clip(resized_heatmap, 0, 1) * 255
        heatmap_colored = cv2.applyColorMap(
            scaled_heatmap.astype(np.uint8), cv2.COLORMAP_JET
        )

        overlay = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)

        gradcam_dir = Path(self.config.media_root) / "gradcam"
        gradcam_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{uuid4().hex}.png"
        output_path = gradcam_dir / filename

        success = cv2.imwrite(str(output_path), overlay)
        if not success:
            return None

        media_url = self.config.media_url.rstrip("/")
        if media_url == "":
            media_url = ""
        return f"{media_url}/gradcam/{filename}"
