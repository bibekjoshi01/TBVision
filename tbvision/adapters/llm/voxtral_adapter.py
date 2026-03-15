import asyncio
from typing import Any, Optional

from mistralai import Mistral

from tbvision.adapters.llm.base import LLMAdapter


class MistralVoxtralAgentAdapter(LLMAdapter):
    def __init__(self, api_key: str, agent_name: str = "voxtral"):
        self.client = Mistral(api_key=api_key)
        voxtral = getattr(self.client, "voxtral", None)
        if voxtral is None:
            raise RuntimeError("Mistral client does not expose the voxtral agent API.")
        self.agent = voxtral.agent(agent_name)
        self.agent_name = agent_name

    def _extract_content(self, response: Any) -> str:
        if not response:
            raise RuntimeError("Voxtral agent returned no response.")
        if hasattr(response, "choices") and response.choices:
            message = getattr(response.choices[0], "message", None)
            if message and getattr(message, "content", None):
                return message.content
        if hasattr(response, "content"):
            return response.content
        return str(response)

    async def generate_text(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> str:
        """
        Run the Voxtral agent pipeline in a thread so it does not block the event loop.
        """

        def _run():
            return self.agent.complete(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        response = await asyncio.to_thread(_run)
        return self._extract_content(response)
