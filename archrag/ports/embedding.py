"""Port: text embedding."""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingPort(ABC):
    """Convert text to dense vectors."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single piece of text."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts (may be optimised by adapter)."""

    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of the embedding space."""
