"""LLM adapter: Ollama local server."""

from __future__ import annotations

import json
from typing import Any

from archrag.ports.llm import LLMPort


class OllamaLLM(LLMPort):
    """Calls the Ollama /api/chat endpoint."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
    ):
        from ollama import Client  # lazy

        self._client = Client(host=base_url)
        self._model = model
        self._temperature = temperature

    def generate(self, prompt: str, *, system: str = "") -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat(
            model=self._model,
            messages=messages,
            options={"temperature": self._temperature},
        )
        return resp["message"]["content"]

    def generate_json(self, prompt: str, *, system: str = "") -> dict[str, Any]:
        raw = self.generate(prompt, system=system)
        # Try to extract JSON from the response
        raw = raw.strip()
        # Find first { and last }
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            raw = raw[start : end + 1]
        return json.loads(raw)
