"""Port: persistent document / metadata storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from archrag.domain.models import Community, CommunityHierarchy, TextChunk


class DocumentStorePort(ABC):
    """Store corpus chunks, community summaries, and hierarchy metadata."""

    # ── chunks ──

    @abstractmethod
    def save_chunk(self, chunk: TextChunk) -> None: ...

    @abstractmethod
    def save_chunks(self, chunks: list[TextChunk]) -> None: ...

    @abstractmethod
    def get_chunk(self, chunk_id: str) -> TextChunk | None: ...

    @abstractmethod
    def get_all_chunks(self) -> list[TextChunk]: ...

    # ── communities ──

    @abstractmethod
    def save_community(self, community: Community) -> None: ...

    @abstractmethod
    def save_communities(self, communities: list[Community]) -> None: ...

    @abstractmethod
    def get_community(self, community_id: str) -> Community | None: ...

    @abstractmethod
    def get_communities_at_level(self, level: int) -> list[Community]: ...

    # ── hierarchy ──

    @abstractmethod
    def save_hierarchy(self, hierarchy: CommunityHierarchy) -> None: ...

    @abstractmethod
    def load_hierarchy(self) -> CommunityHierarchy | None: ...

    # ── generic key/value (for index metadata etc.) ──

    @abstractmethod
    def put_meta(self, key: str, value: Any) -> None: ...

    @abstractmethod
    def get_meta(self, key: str) -> Any: ...

    # ── delete ──

    @abstractmethod
    def delete_chunk(self, chunk_id: str) -> None:
        """Remove a single chunk."""

    @abstractmethod
    def search_chunks(self, query: str) -> list[TextChunk]:
        """Case-insensitive substring search on chunk text."""

    # ── lifecycle ──

    @abstractmethod
    def clear(self) -> None: ...

    def clone(self) -> "DocumentStorePort":
        """Create an independent deep copy of this store (for blue/green swap)."""
        raise NotImplementedError(f"{type(self).__name__} does not support clone()")
