from openai import OpenAI
from app.config import settings
from typing import List, Dict


class LLMClient:
    """
    A thin wrapper around the LLM API using the OpenAI SDK.

    Why this design (Strategy / Adapter Pattern)?
    Groq is OpenAI-API-compatible, so we use the same openai SDK
    but point it at Groq's base_url. To swap providers, we only
    need to change two config values (api_key + base_url).
    This makes the LLM completely swappable without touching any
    other part of the codebase — a key interview talking point.
    """

    def __init__(self) -> None:
        self._client = OpenAI(
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
        )
        self._model = settings.llm_model

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 400,
    ) -> str:
        """
        Sends a list of messages to the LLM and returns the reply as a string.

        Args:
            messages: Standard OpenAI message format.
                      e.g. [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            temperature: Controls randomness. Lower = more deterministic.
                         We use 0.2 for structured JSON responses (routing),
                         slightly higher for answer generation.
            max_tokens: Limits output token generation to avoid rate limits.

        Returns:
            The assistant's reply as a plain string.
        """
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
