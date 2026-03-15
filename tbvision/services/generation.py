"""Helper that drives LLM-based explanations for the analyzer workflow."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List

from tbvision.adapters.llm.gemini_adapter import GeminiAdapter
from tbvision.adapters.llm.mistral_adapter import MistralAdapter
from tbvision.adapters.llm.voxtral_adapter import MistralVoxtralAgentAdapter
from tbvision.core.config import Settings
from tbvision.prompts.explanation import explanation_prompt, synthesis_prompt
from tbvision.prompts.followup import followup_prompt

logger = logging.getLogger(__name__)


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
        self.mistral_voxtral: MistralVoxtralAgentAdapter | None = None
        if settings.mistral_api_key:
            try:
                self.mistral_voxtral = MistralVoxtralAgentAdapter(
                    settings.mistral_api_key.get_secret_value()
                )
            except Exception as exc:
                logger.warning(
                    "Could not initialize Mistral Voxtral agent (%s); falling back to chat",
                    exc,
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

        return await adapter.generate_text(prompt, temperature=0.2, max_tokens=2000)

    async def generate_followup(
        self,
        report_context: Dict[str, Any],
        question: str,
        history: Iterable[Dict[str, str]],
    ) -> str:
        adapter = self.mistral_voxtral or self.mistral
        if not adapter:
            return "Follow-up generation is unavailable (no LLM configured)."

        prompt = self._build_followup_prompt(report_context, question, history)
        return await adapter.generate_text(prompt, temperature=0.3, max_tokens=1200)

    def _build_followup_prompt(
        self,
        report_context: Dict[str, Any],
        question: str,
        history: Iterable[Dict[str, str]],
    ) -> str:
        metadata = report_context.get("metadata") or {}
        analysis = report_context.get("analysis") or {}

        metadata_summary = self._summarize_metadata(metadata)
        stored_summary = report_context.get("summary")
        analysis_summary = "\n".join(
            part
            for part in [
                stored_summary,
                self._summarize_analysis(analysis),
            ]
            if part
        )
        evidence_summary = report_context.get(
            "evidence_summary"
        ) or self._summarize_evidence(analysis.get("evidence", []))
        uncertainty_note = self._build_uncertainty_note(report_context, analysis)
        history_text = self._format_history(history)

        return followup_prompt.format(
            metadata_summary=metadata_summary,
            analysis_summary=analysis_summary,
            evidence_summary=evidence_summary,
            uncertainty_note=uncertainty_note,
            history=history_text,
            question=question,
        )

    def _summarize_metadata(self, metadata: dict[str, Any]) -> str:
        lines = []
        for key, value in metadata.items():
            if not value:
                continue
            if isinstance(value, dict):
                inner = ", ".join(
                    f"{sub_key}={sub_value}"
                    for sub_key, sub_value in value.items()
                    if sub_value is not None
                )
                if inner:
                    lines.append(f"{key}: {inner}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines) or "Not provided."

    def _summarize_analysis(self, analysis: dict[str, Any]) -> str:
        prob = analysis.get("probability")
        prob_display = f"{prob:.2f}" if isinstance(prob, (int, float)) else "N/A"
        std = analysis.get("uncertainty_std")
        std_display = f"{std:.2f}" if isinstance(std, (int, float)) else "N/A"
        lines = [
            f"Prediction: {analysis.get('prediction', 'Unknown')} ({analysis.get('prediction_label', 'N/A')}), probability={prob_display}",
            f"Uncertainty: {analysis.get('uncertainty_level', 'Unknown')} (std={std_display})",
            f"Grad-CAM region: {analysis.get('gradcam_region', 'unspecified')}",
            f"Explanation: {analysis.get('explanation', 'None')}",
        ]
        return "\n".join(lines)

    def _summarize_evidence(self, evidence_list: list[dict[str, Any]]) -> str:
        if not evidence_list:
            return "No retrieval evidence was available."
        excerpts = []
        for entry in evidence_list[:3]:
            text = entry.get("text", "").replace("\n", " ").strip()
            header = (
                entry.get("metadata", {}).get("title") or entry.get("id") or "Evidence"
            )
            if text:
                excerpts.append(f"{header}: {text}")
        return "\n".join(excerpts) or "No concise evidence text found."

    def _format_history(self, history: Iterable[Dict[str, str]]) -> str:
        if not history:
            return "None"
        lines = [
            f"Q: {entry['question']}\nA: {entry['answer']}"
            for entry in history
            if entry.get("question") and entry.get("answer")
        ]
        return "\n".join(lines) or "None"

    def _build_uncertainty_note(
        self,
        report_context: Dict[str, Any],
        analysis: dict[str, Any],
    ) -> str:
        level = report_context.get("uncertainty_level") or analysis.get(
            "uncertainty_level"
        )
        std = report_context.get("uncertainty_std") or analysis.get("uncertainty_std")
        std_display = f"{std:.2f}" if isinstance(std, (int, float)) else "N/A"
        note_parts = []
        if level:
            note_parts.append(f"Reported uncertainty level: {level}.")
        else:
            note_parts.append("Uncertainty level not explicitly reported.")
        note_parts.append(f"Standard deviation: {std_display}.")
        if level == "High" or (isinstance(std, (int, float)) and std >= 0.2):
            note_parts.append(
                "Uncertainty is elevated; prioritize confirmatory diagnostics or specialist review."
            )
        else:
            note_parts.append(
                "Uncertainty is within tolerable bounds; proceed with standard follow-up."
            )
        return " ".join(note_parts)

    @staticmethod
    def summarize_evidence(documents: Iterable[Dict[str, Any]]) -> str:
        items: List[str] = []
        for doc in documents:
            header = doc.get("metadata", {}).get("title") or doc.get("id") or "Evidence"
            text = doc.get("text", "").strip()
            if text:
                items.append(f"{header}: {text}")
        return "\n".join(items)
