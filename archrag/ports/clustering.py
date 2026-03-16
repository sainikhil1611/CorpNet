"""Port: graph clustering algorithm."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class WeightedEdge:
    """An edge with a similarity weight for clustering."""

    source: str
    target: str
    weight: float


class ClusteringPort(ABC):
    """Detect communities in a weighted graph."""

    @abstractmethod
    def cluster(
        self,
        node_ids: list[str],
        edges: list[WeightedEdge],
    ) -> list[list[str]]:
        """Return a list of communities, each a list of node IDs."""
