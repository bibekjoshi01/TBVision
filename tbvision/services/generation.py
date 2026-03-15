"""Helper that drives LLM-based explanations for the analyzer workflow."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from tbvision.adapters.llm.gemini_adapter import GeminiAdapter
from tbvision.adapters.llm.mistral_adapter import MistralAdapter
from tbvision.core.config import Settings
from tbvision.prompts.explanation import explanation_prompt, synthesis_prompt


class GenerationService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.gemini = (
            GeminiAdapter(settings.gemini_api_key.get_secret_value())
            if settings.gemini_api_key
            else None
        )
        self.mistral = (
            MistralAdapter(settings.mistral_api_key.get_secret_value())
            if settings.mistral_api_key
            else None
        )

    async def generate_validation(
        self,
        prediction_label: str,
        probability: float,
        region: str,
        patient_info: str,
    ) -> str:
        if not self.gemini:
            return "Gemini validation skipped (API key not configured)."

        prompt = explanation_prompt.format(
            prediction_label=prediction_label,
            prob=probability,
            region=region,
            patient_info=patient_info,
        )

        return await self.gemini.generate_text(prompt, temperature=0.2, max_tokens=450)

    async def generate_synthesis(
        self,
        prediction_label: str,
        probability: float,
        uncertainty: str,
        uncertainty_std: float,
        region: str,
        patient_info: str,
        evidence_text: str,
        gemini_validation: str,
    ) -> str:
        adapter = self.mistral or self.gemini
        if not adapter:
            return "Clinical synthesis unavailable (no LLM configured)."

        prompt = synthesis_prompt.format(
            prediction_label=prediction_label,
            prob=probability,
            uncertainty=uncertainty,
            uncertainty_std=uncertainty_std,
            region=region,
            gemini_validation=gemini_validation,
            patient_info=patient_info,
            evidence_text=evidence_text or "No WHO evidence retrieved.",
        )

        return await adapter.generate_text(prompt, temperature=0.2, max_tokens=1200)

    @staticmethod
    def summarize_evidence(documents: Iterable[Dict[str, Any]]) -> str:
        items: List[str] = []
        for doc in documents:
            header = doc.get("metadata", {}).get("title") or doc.get("id") or "Evidence"
            text = doc.get("text", "").strip()
            if text:
                items.append(f"{header}: {text}")
        return "\n".join(items)
