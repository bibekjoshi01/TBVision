import asyncio

from mistralai import Mistral

from app.adapters.llm.base import LLMAdapter


class MistralAdapter(LLMAdapter):
    def __init__(self, api_key: str):
        self.client = Mistral(api_key=api_key)
        self.model_name = "mistral-large-latest"

    async def generate_text(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1200,
    ) -> str:
        """
        Generate text using Mistral with controlled parameters.
        """

        def _generate():
            return self.client.chat.complete(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )

        try:
            response = await asyncio.to_thread(_generate)

            if (
                not response
                or not response.choices
                or not response.choices[0].message
                or not response.choices[0].message.content
            ):
                raise RuntimeError("Mistral returned empty response")

            return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"Mistral generation failed: {str(e)}")
