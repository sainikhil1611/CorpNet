"""Pure domain models — zero external dependencies."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


def _uid() -> str:
    return uuid.uuid4().hex[:12]


# ── Corpus ──────────────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    """A segment of a source document."""

    text: str
    id: str = field(default_factory=_uid)
    source_doc: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Knowledge Graph ─────────────────────────────────────────────────────────

@dataclass
class Entity:
    """A node in the knowledge graph."""

    name: str
    description: str
    id: str = field(default_factory=_uid)
    entity_type: str = ""
    source_chunk_ids: list[str] = field(default_factory=list)
    embedding: list[float] | None = None


@dataclass
class Relation:
    """An edge in the knowledge graph."""

    source_id: str
    target_id: str
    description: str
    id: str = field(default_factory=_uid)
    weight: float = 1.0
    source_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class KnowledgeGraph:
    """Container for entities and relations."""

    entities: dict[str, Entity] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)

    # ── helpers ──

    def add_entity(self, entity: Entity) -> None:
        self.entities[entity.id] = entity

    def add_relation(self, relation: Relation) -> None:
        self.relations.append(relation)

    def neighbours(self, entity_id: str) -> list[str]:
        """Return IDs of entities connected to *entity_id*."""
        ids: list[str] = []
        for r in self.relations:
            if r.source_id == entity_id:
                ids.append(r.target_id)
            elif r.target_id == entity_id:
                ids.append(r.source_id)
        return ids

    def entity_by_name(self, name: str) -> Entity | None:
        for e in self.entities.values():
            if e.name.lower() == name.lower():
                return e
        return None


# ── Attributed Communities ──────────────────────────────────────────────────

@dataclass
class Community:
    """An attributed community at a single hierarchical level."""

    id: str = field(default_factory=_uid)
    level: int = 0
    member_ids: list[str] = field(default_factory=list)  # entity or child-community ids
    summary: str = ""
    embedding: list[float] | None = None
    parent_id: str | None = None


@dataclass
class CommunityHierarchy:
    """The full multi-level hierarchy Δ of attributed communities."""

    levels: list[list[Community]] = field(default_factory=list)  # index = level

    @property
    def height(self) -> int:
        return len(self.levels)

    def communities_at(self, level: int) -> list[Community]:
        if 0 <= level < len(self.levels):
            return self.levels[level]
        return []

    def all_communities(self) -> list[Community]:
        return [c for level in self.levels for c in level]


# ── C-HNSW Index ────────────────────────────────────────────────────────────

@dataclass
class CHNSWNode:
    """A node inside the C-HNSW index."""

    id: str
    level: int
    embedding: list[float]
    label: str = ""  # entity name or community summary snippet
    intra_neighbours: list[str] = field(default_factory=list)  # same-layer links
    inter_link_down: str | None = None  # nearest neighbour in layer below


@dataclass
class CHNSWIndex:
    """Community-based Hierarchical Navigable Small World index."""

    nodes: dict[str, CHNSWNode] = field(default_factory=dict)  # id → node
    layers: list[list[str]] = field(default_factory=list)  # layer i → list of node ids
    M: int = 32  # max intra-layer connections per node
    ef_construction: int = 100

    @property
    def height(self) -> int:
        return len(self.layers)

    def nodes_at(self, level: int) -> list[CHNSWNode]:
        if 0 <= level < len(self.layers):
            return [self.nodes[nid] for nid in self.layers[level]]
        return []


# ── Search / Query results ──────────────────────────────────────────────────

@dataclass
class SearchResult:
    """A single retrieved element from the C-HNSW."""

    node_id: str
    level: int
    distance: float
    text: str = ""


@dataclass
class AnalysisPoint:
    """One point from the LLM's adaptive filtering report."""

    description: str
    score: float


@dataclass
class AnalysisReport:
    """Result of LLM adaptive filtering for one layer."""

    level: int
    points: list[AnalysisPoint] = field(default_factory=list)
