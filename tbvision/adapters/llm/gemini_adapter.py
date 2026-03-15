import asyncio

from google import genai  # type: ignore[import, attr-defined]

from app.adapters.llm.base import LLMAdapter


class GeminiAdapter(LLMAdapter):
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.5-flash"

    async def generate_text(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1200,
    ) -> str:
        """
        Generate text using Gemini with controlled parameters.
        """

        def _generate():
            return self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
            )

        try:
            response = await asyncio.to_thread(_generate)

            if not response or not response.text:
                raise RuntimeError("Gemini returned empty response")

            return response.text

        except Exception as e:
            raise RuntimeError(f"Gemini generation failed: {str(e)}")
