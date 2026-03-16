"""Graph store adapter: in-memory (for tests)."""

from __future__ import annotations

from archrag.domain.models import Entity, Relation
from archrag.ports.graph_store import GraphStorePort


class InMemoryGraphStore(GraphStorePort):
    """Non-persistent graph store — useful for unit tests."""

    def __init__(self) -> None:
        self._entities: dict[str, Entity] = {}
        self._relations: list[Relation] = []

    def save_entity(self, entity: Entity) -> None:
        self._entities[entity.id] = entity

    def save_entities(self, entities: list[Entity]) -> None:
        for e in entities:
            self._entities[e.id] = e

    def save_relation(self, relation: Relation) -> None:
        self._relations.append(relation)

    def save_relations(self, relations: list[Relation]) -> None:
        self._relations.extend(relations)

    def get_entity(self, entity_id: str) -> Entity | None:
        return self._entities.get(entity_id)

    def get_all_entities(self) -> list[Entity]:
        return list(self._entities.values())

    def get_entity_by_name(self, name: str) -> Entity | None:
        for e in self._entities.values():
            if e.name.lower() == name.lower():
                return e
        return None

    def get_relations_for(self, entity_id: str) -> list[Relation]:
        return [
            r
            for r in self._relations
            if r.source_id == entity_id or r.target_id == entity_id
        ]

    def get_all_relations(self) -> list[Relation]:
        return list(self._relations)

    def get_neighbours(self, entity_id: str) -> list[Entity]:
        ids: set[str] = set()
        for r in self._relations:
            if r.source_id == entity_id:
                ids.add(r.target_id)
            elif r.target_id == entity_id:
                ids.add(r.source_id)
        return [self._entities[eid] for eid in ids if eid in self._entities]

    def clear(self) -> None:
        self._entities.clear()
        self._relations.clear()

    def delete_entity(self, entity_id: str) -> None:
        self._entities.pop(entity_id, None)
        self._relations = [
            r for r in self._relations
            if r.source_id != entity_id and r.target_id != entity_id
        ]

    def search_entities_by_name(self, query: str) -> list[Entity]:
        q = query.lower()
        return [e for e in self._entities.values() if q in e.name.lower()]

    def clone(self) -> "InMemoryGraphStore":
        """Create an independent deep copy."""
        new = InMemoryGraphStore()
        new._entities = dict(self._entities)
        new._relations = list(self._relations)
        return new
