import logging
from typing import Any, Dict, Optional

from tbvision.core.config import Settings
from tbvision.services.classifier import ClassifierService
from tbvision.services.generation import GenerationService
from tbvision.services.retrieval import RetrievalService
from tbvision.utils.check_internet import check_internet_connection

logger = logging.getLogger(__name__)


class Analyzer:
    def __init__(
        self,
        image,
        settings: Settings,
        classifier_service: ClassifierService,
        retrieval_service: RetrievalService,
        generation_service: GenerationService,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.image = image
        self.metadata = metadata or {}
        self.settings = settings
        self.classifier_service = classifier_service
        self.retrieval_service = retrieval_service
        self.generation_service = generation_service
        self.is_online = check_internet_connection()

    async def analyze(self):
        gradcam_data: Dict[str, Any] = {}
        gradcam_image: Optional[str] = None

        try:
            logger.debug("Running classifier prediction for analyzer pipeline.")
            pred_data = self.classifier_service.predict(self.image)
            gradcam_data = self.classifier_service.analyze_gradcam(pred_data)
            logger.debug(
                "Grad-CAM regions extracted: %s", gradcam_data.get("dominant_region")
            )
            try:
                gradcam_image = self.classifier_service.create_gradcam_overlay(
                    self.image, gradcam_data["heatmap"]
                )
            except Exception as e:
                logger.error("Grad-CAM overlay failed: %s", e, exc_info=True)
                raise ValueError("Grad-CAM++ overlay generation failed.") from e

        except Exception as err:
            logger.error("Analysis failed: %s", err, exc_info=True)
            raise ValueError("Error occurred while running analyzer.") from err

        # Make Explaination Report If Internet
        if self.is_online:
            evidence = await self.retrieve_evidence(
                pred_data["probability"], gradcam_data["dominant_region"]
            )
            pred_data["evidence"] = evidence

            try:
                explanation = await self.generate_explanation(pred_data, gradcam_data)
            except Exception as e:
                logger.error("Explanation generation failed: %s", e, exc_info=True)
                explanation = "Unable to generate explanation."
        else:
            pred_data["evidence"] = []
            explanation = "Online services unavailable."

        pred_data["gradcam_region"] = gradcam_data.get("description")
        pred_data["gradcam_image"] = gradcam_image
        pred_data["explanation"] = explanation

        pred_data.pop("image_tensor", None)
        return pred_data

    async def generate_explanation(
        self, pred_data: Dict[str, Any], gradcam_data: Dict[str, Any]
    ) -> str:
        evidence_text = self.generation_service.summarize_evidence(
            pred_data.get("evidence", [])
        )
        patient_info = self._format_metadata()

        gemini_validation = await self.generation_service.generate_validation(
            pred_data["prediction_label"],
            pred_data["probability"],
            gradcam_data.get("description", "unspecified region"),
            patient_info,
        )

        synthesis = await self.generation_service.generate_synthesis(
            pred_data["prediction_label"],
            pred_data["probability"],
            pred_data.get("uncertainty_level", "Unknown"),
            pred_data.get("uncertainty_std", 0.0),
            gradcam_data.get("description", "unspecified region"),
            patient_info,
            evidence_text,
            gemini_validation,
        )

        return synthesis

    async def retrieve_evidence(self, prediction, region, threshold=0.5):
        """Retrieve medical evidence from the vector DB when TB is suspected."""

        if prediction < threshold or not self.retrieval_service.available():
            return []

        query = (
            f"Pulmonary tuberculosis chest x-ray findings near the {region} lung zones "
            "with consolidation, cavitation, or nodular features referencing WHO imaging guidance."
        )

        try:
            return await self.retrieval_service.retrieve(
                query, top_k=self.settings.top_k, filters={}
            )
        except Exception as exc:
            logger.error("Retrieval failure: %s", exc, exc_info=True)
            return []

    def _format_metadata(self) -> str:
        if not self.metadata:
            return "Not provided."

        metadata = (
            self.metadata.dict()
            if hasattr(self.metadata, "dict")
            else dict(self.metadata)
        )
        parts = []
        for key, value in metadata.items():
            if not value:
                continue
            if isinstance(value, dict):
                inner = ", ".join(f"{k}={v}" for k, v in value.items() if v is not None)
                if inner:
                    parts.append(f"{key}: {inner}")
            else:
                parts.append(f"{key}: {value}")
        
        return "; ".join(parts) or "Not provided."
