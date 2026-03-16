"""Port: persistent knowledge-graph storage."""

from __future__ import annotations

from abc import ABC, abstractmethod

from archrag.domain.models import Entity, Relation


class GraphStorePort(ABC):
    """Persist and query KG entities + relations."""

    # ── write ──

    @abstractmethod
    def save_entity(self, entity: Entity) -> None: ...

    @abstractmethod
    def save_entities(self, entities: list[Entity]) -> None: ...

    @abstractmethod
    def save_relation(self, relation: Relation) -> None: ...

    @abstractmethod
    def save_relations(self, relations: list[Relation]) -> None: ...

    # ── read ──

    @abstractmethod
    def get_entity(self, entity_id: str) -> Entity | None: ...

    @abstractmethod
    def get_all_entities(self) -> list[Entity]: ...

    @abstractmethod
    def get_entity_by_name(self, name: str) -> Entity | None: ...

    @abstractmethod
    def get_relations_for(self, entity_id: str) -> list[Relation]: ...

    @abstractmethod
    def get_all_relations(self) -> list[Relation]: ...

    @abstractmethod
    def get_neighbours(self, entity_id: str) -> list[Entity]: ...

    # ── delete ──

    @abstractmethod
    def delete_entity(self, entity_id: str) -> None:
        """Delete an entity and all its relations."""

    @abstractmethod
    def search_entities_by_name(self, query: str) -> list[Entity]:
        """Case-insensitive substring search on entity names."""

    # ── lifecycle ──

    @abstractmethod
    def clear(self) -> None: ...

    def clone(self) -> "GraphStorePort":
        """Create an independent deep copy of this store (for blue/green swap)."""
        raise NotImplementedError(f"{type(self).__name__} does not support clone()")
