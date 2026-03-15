from abc import ABC, abstractmethod


class LLMAdapter(ABC):
    @abstractmethod
    async def generate_text(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        pass
