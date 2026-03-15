import asyncio

from transformers import pipeline  # type: ignore[import]

from app.adapters.llm.base import LLMAdapter


class LocalAdapter(LLMAdapter):
    def __init__(self, model_name: str):
        self.generator = pipeline("text-generation", model=model_name)

    async def generate_text(self, prompt: str, temperature: float, max_tokens: int) -> str:
        result = await asyncio.to_thread(
            self.generator,
            prompt,
            max_length=max_tokens,
            temperature=temperature,
            num_return_sequences=1,
        )
        return result[0]["generated_text"]
