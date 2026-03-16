"""LLM adapter: OpenAI API."""

from __future__ import annotations

import json
from typing import Any

from archrag.ports.llm import LLMPort


class OpenAILLM(LLMPort):
    """Uses the OpenAI chat completions endpoint."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.0,
    ):
        from openai import OpenAI  # lazy

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._temperature = temperature

    def generate(self, prompt: str, *, system: str = "") -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self._temperature,
        )
        return resp.choices[0].message.content or ""

    def generate_json(self, prompt: str, *, system: str = "") -> dict[str, Any]:
        raw = self.generate(prompt, system=system)
        raw = raw.strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            raw = raw[start : end + 1]
        return json.loads(raw)
