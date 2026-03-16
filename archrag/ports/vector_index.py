"""Port: approximate nearest-neighbour vector index (C-HNSW backing store)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class VectorIndexPort(ABC):
    """Store and search dense vectors, organised by layer."""

    @abstractmethod
    def add_vectors(
        self,
        ids: list[str],
        vectors: np.ndarray,
        *,
        layer: int = 0,
    ) -> None:
        """Insert vectors into the given layer."""

    @abstractmethod
    def search(
        self,
        query: np.ndarray,
        k: int,
        *,
        layer: int = 0,
        candidate_ids: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Return up to *k* (id, distance) pairs from *layer*."""

    @abstractmethod
    def get_vector(self, id: str) -> np.ndarray | None:
        """Return stored vector for *id* or ``None``."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist index to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore index from disk."""

    @abstractmethod
    def clear(self) -> None: ...

    def clone(self) -> "VectorIndexPort":
        """Create an independent deep copy of this index (for blue/green swap)."""
        raise NotImplementedError(f"{type(self).__name__} does not support clone()")
