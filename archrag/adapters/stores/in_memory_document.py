"""Document store adapter: in-memory (for tests)."""

from __future__ import annotations

import json
from typing import Any

from archrag.domain.models import Community, CommunityHierarchy, TextChunk
from archrag.ports.document_store import DocumentStorePort


class InMemoryDocumentStore(DocumentStorePort):
    """Non-persistent doc store — for unit tests."""

    def __init__(self) -> None:
        self._chunks: dict[str, TextChunk] = {}
        self._communities: dict[str, Community] = {}
        self._meta: dict[str, Any] = {}

    # ── chunks ──

    def save_chunk(self, chunk: TextChunk) -> None:
        self._chunks[chunk.id] = chunk

    def save_chunks(self, chunks: list[TextChunk]) -> None:
        for c in chunks:
            self._chunks[c.id] = c

    def get_chunk(self, chunk_id: str) -> TextChunk | None:
        return self._chunks.get(chunk_id)

    def get_all_chunks(self) -> list[TextChunk]:
        return list(self._chunks.values())

    # ── communities ──

    def save_community(self, community: Community) -> None:
        self._communities[community.id] = community

    def save_communities(self, communities: list[Community]) -> None:
        for c in communities:
            self._communities[c.id] = c

    def get_community(self, community_id: str) -> Community | None:
        return self._communities.get(community_id)

    def get_communities_at_level(self, level: int) -> list[Community]:
        return [c for c in self._communities.values() if c.level == level]

    # ── hierarchy ──

    def save_hierarchy(self, hierarchy: CommunityHierarchy) -> None:
        for level_comms in hierarchy.levels:
            self.save_communities(level_comms)
        structure = {
            "height": hierarchy.height,
            "level_ids": [
                [c.id for c in level_comms] for level_comms in hierarchy.levels
            ],
        }
        self.put_meta("hierarchy_structure", json.dumps(structure))

    def load_hierarchy(self) -> CommunityHierarchy | None:
        raw = self.get_meta("hierarchy_structure")
        if raw is None:
            return None
        structure = json.loads(raw)
        levels: list[list[Community]] = []
        for id_list in structure["level_ids"]:
            comms = [self.get_community(cid) for cid in id_list]
            levels.append([c for c in comms if c is not None])
        return CommunityHierarchy(levels=levels)

    # ── meta ──

    def put_meta(self, key: str, value: Any) -> None:
        self._meta[key] = value

    def get_meta(self, key: str) -> Any:
        return self._meta.get(key)

    def clear(self) -> None:
        self._chunks.clear()
        self._communities.clear()
        self._meta.clear()

    def delete_chunk(self, chunk_id: str) -> None:
        self._chunks.pop(chunk_id, None)

    def search_chunks(self, query: str) -> list[TextChunk]:
        q = query.lower()
        return [c for c in self._chunks.values() if q in c.text.lower()]

    def clone(self) -> "InMemoryDocumentStore":
        """Create an independent deep copy."""
        new = InMemoryDocumentStore()
        new._chunks = dict(self._chunks)
        new._communities = dict(self._communities)
        new._meta = dict(self._meta)
        return new
