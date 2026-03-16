"""Port: large language model inference."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMPort(ABC):
    """Generate text completions from prompts."""

    @abstractmethod
    def generate(self, prompt: str, *, system: str = "") -> str:
        """Return a free-form text completion."""

    @abstractmethod
    def generate_json(self, prompt: str, *, system: str = "") -> dict[str, Any]:
        """Return a structured JSON completion (parsed)."""
