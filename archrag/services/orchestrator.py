"""Orchestrator: wires offline indexing and online retrieval pipelines.

Uses a blue/green snapshot pattern for lock-free reads during reindex.
All mutable state lives inside a ``_Snapshot``.  Writers clone the
adapters, build into the shadow copy, then atomically swap
``self._snapshot``.  Python's GIL guarantees that reference assignment
is atomic, so readers never see a half-built state.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from archrag.domain.models import CHNSWIndex, CommunityHierarchy, KnowledgeGraph
from archrag.ports.clustering import ClusteringPort
from archrag.ports.document_store import DocumentStorePort
from archrag.ports.embedding import EmbeddingPort
from archrag.ports.graph_store import GraphStorePort
from archrag.ports.llm import LLMPort
from archrag.ports.vector_index import VectorIndexPort
from archrag.services.adaptive_filtering import AdaptiveFilteringService
from archrag.services.chnsw_build import CHNSWBuildService
from archrag.services.hierarchical_clustering import HierarchicalClusteringService
from archrag.services.hierarchical_search import HierarchicalSearchService
from archrag.services.kg_construction import KGConstructionService

log = logging.getLogger(__name__)


# ── Immutable snapshot of all mutable state ──────────────────────────


@dataclass(frozen=True)
class _Snapshot:
    """A consistent, point-in-time view of the datastore.

    Readers grab ``self._snapshot`` once at method entry and work
    entirely against that reference.  Writers build a new snapshot
    and atomically replace ``self._snapshot``.
    """

    graph_store: GraphStorePort
    doc_store: DocumentStorePort
    vector_index: VectorIndexPort
    chnsw_index: CHNSWIndex | None

    # Sub-services wired to *these* adapters
    kg_service: KGConstructionService
    cluster_service: HierarchicalClusteringService
    chnsw_service: CHNSWBuildService
    search_service: HierarchicalSearchService


class ArchRAGOrchestrator:
    """Top-level entry point for the ArchRAG pipeline.

    Uses a **blue/green snapshot swap** so that reads (``query``,
    ``search_*``, ``stats``) are never blocked by writes (``index``,
    ``add_documents``).

    * Readers snapshot ``self._snapshot`` at the top of every call.
    * Writers clone the mutable adapters, build into the shadow,
      then do ``self._snapshot = new_snap`` — an atomic reference
      swap under the GIL.
    * A ``_write_lock`` serialises concurrent writers (only one
      reindex at a time), but readers never touch it.
    """

    def __init__(
        self,
        llm: LLMPort,
        embedding: EmbeddingPort,
        graph_store: GraphStorePort,
        doc_store: DocumentStorePort,
        vector_index: VectorIndexPort,
        clustering: ClusteringPort,
        *,
        chunk_size: int = 1200,
        chunk_overlap: int = 100,
        max_levels: int = 5,
        similarity_threshold: float = 0.7,
        M: int = 32,
        ef_construction: int = 100,
        k_per_layer: int = 5,
    ):
        # Immutable / shared (stateless services and config)
        self._llm = llm
        self._embedding = embedding
        self._clustering = clustering

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._max_levels = max_levels
        self._similarity_threshold = similarity_threshold
        self._M = M
        self._ef_construction = ef_construction
        self._k_per_layer = k_per_layer

        self._filter_service = AdaptiveFilteringService(llm)

        # Serialise concurrent writers
        self._write_lock = threading.Lock()

        # Build the initial (blue) snapshot
        self._snapshot = self._build_snapshot(
            graph_store, doc_store, vector_index, chnsw_index=None,
        )

    # ── snapshot helpers ──

    def _build_snapshot(
        self,
        graph_store: GraphStorePort,
        doc_store: DocumentStorePort,
        vector_index: VectorIndexPort,
        *,
        chnsw_index: CHNSWIndex | None,
    ) -> _Snapshot:
        """Construct a ``_Snapshot`` with sub-services wired to the given adapters."""
        return _Snapshot(
            graph_store=graph_store,
            doc_store=doc_store,
            vector_index=vector_index,
            chnsw_index=chnsw_index,
            kg_service=KGConstructionService(
                self._llm, self._embedding, graph_store, doc_store,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            ),
            cluster_service=HierarchicalClusteringService(
                self._llm, self._embedding, self._clustering,
                graph_store, doc_store,
                max_levels=self._max_levels,
                similarity_threshold=self._similarity_threshold,
            ),
            chnsw_service=CHNSWBuildService(
                self._embedding, vector_index, graph_store, doc_store,
                M=self._M,
                ef_construction=self._ef_construction,
            ),
            search_service=HierarchicalSearchService(
                self._embedding, vector_index, graph_store, doc_store,
                k_per_layer=self._k_per_layer,
            ),
        )

    def _clone_snapshot_adapters(
        self,
        snap: _Snapshot,
    ) -> tuple[GraphStorePort, DocumentStorePort, VectorIndexPort]:
        """Clone the mutable adapters from an existing snapshot."""
        return (
            snap.graph_store.clone(),
            snap.doc_store.clone(),
            snap.vector_index.clone(),
        )

    # ── Offline indexing (writer) ──

    def index(self, corpus_path: str) -> None:
        """Run the full offline indexing pipeline (blue/green).

        1. Load corpus (JSONL)
        2. Clone adapters → shadow
        3. Clear shadow stores
        4. KG construction → shadow
        5. Hierarchical clustering → shadow
        6. C-HNSW build → shadow
        7. Atomic swap
        """
        log.info("Starting offline indexing from %s", corpus_path)
        documents = self._load_corpus(corpus_path)
        log.info("Loaded %d documents", len(documents))

        with self._write_lock:
            # Clone current adapters into a shadow set
            gs, ds, vi = self._clone_snapshot_adapters(self._snapshot)

            # Fresh index: wipe shadow stores
            gs.clear()
            ds.clear()
            vi.clear()

            shadow = self._build_snapshot(gs, ds, vi, chnsw_index=None)

            # Phase 1: KG construction (into shadow stores)
            kg = shadow.kg_service.build(documents)

            # Phase 2: Hierarchical clustering
            hierarchy = shadow.cluster_service.build(kg)

            # Phase 3: C-HNSW construction
            chnsw_index = shadow.chnsw_service.build(hierarchy)

            # Build final snapshot with the index
            shadow = self._build_snapshot(gs, ds, vi, chnsw_index=chnsw_index)

            # ── Atomic swap ──
            self._snapshot = shadow

        log.info("Offline indexing complete.")

    # ── Online retrieval (reader) ──

    def query(self, question: str) -> str:
        """Answer a question using hierarchical search + adaptive filtering."""
        log.info("Query: %s", question)

        # Snapshot: grab the current state once
        snap = self._snapshot

        # Load persisted vector index if not already in memory
        if snap.chnsw_index is None:
            vec_path = Path("data/chnsw_vectors.json")
            if vec_path.exists():
                log.info("Loading vector index from %s", vec_path)
                snap.vector_index.load(str(vec_path))
            else:
                log.warning("No persisted vector index found at %s", vec_path)

        # Hierarchical search
        results = snap.search_service.search(question, snap.chnsw_index)

        total = sum(len(layer) for layer in results)
        log.info("Retrieved %d results across %d layers", total, len(results))

        # Adaptive filtering & answer generation
        answer = self._filter_service.generate_answer(question, results)

        return answer

    # ── Add / Remove / Search ──

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        """Add new documents to an existing index (blue/green).

        Clones the current stores, merges new docs, rebuilds hierarchy
        + C-HNSW on the shadow, then atomically swaps in.
        """
        log.info("Adding %d documents to existing index", len(documents))

        with self._write_lock:
            # Clone current state
            gs, ds, vi = self._clone_snapshot_adapters(self._snapshot)
            shadow = self._build_snapshot(gs, ds, vi, chnsw_index=None)

            # Build KG for new docs (merges into shadow stores)
            kg = shadow.kg_service.build(documents)

            # Rebuild hierarchy + C-HNSW over the full graph in shadow
            all_entities = gs.get_all_entities()
            full_kg = KnowledgeGraph()
            for e in all_entities:
                full_kg.add_entity(e)
            for r in gs.get_all_relations():
                full_kg.add_relation(r)

            hierarchy = shadow.cluster_service.build(full_kg)
            chnsw_index = shadow.chnsw_service.build(hierarchy)

            shadow = self._build_snapshot(gs, ds, vi, chnsw_index=chnsw_index)

            # ── Atomic swap ──
            self._snapshot = shadow

        log.info("Re-indexed with new documents.")

    def remove_entity(self, entity_name: str) -> bool:
        """Remove an entity by name and cascade-delete its relations.

        This is a small mutation — done in-place on the live snapshot.
        Returns True if found and removed, False otherwise.
        """
        snap = self._snapshot
        entity = snap.graph_store.get_entity_by_name(entity_name)
        if entity is None:
            log.warning("Entity not found: %s", entity_name)
            return False
        snap.graph_store.delete_entity(entity.id)
        log.info("Deleted entity '%s' (id=%s) and its relations.", entity_name, entity.id)
        return True

    def search_entities(self, query: str) -> list[dict[str, str]]:
        """Search entities by name substring. Returns a simple dict list."""
        snap = self._snapshot
        entities = snap.graph_store.search_entities_by_name(query)
        return [
            {
                "id": e.id,
                "name": e.name,
                "type": e.entity_type,
                "description": e.description,
            }
            for e in entities
        ]

    def search_chunks(self, query: str) -> list[dict[str, str]]:
        """Search chunks by text substring."""
        snap = self._snapshot
        chunks = snap.doc_store.search_chunks(query)
        return [
            {
                "id": c.id,
                "text": c.text[:200] + ("..." if len(c.text) > 200 else ""),
                "source": c.source_doc,
            }
            for c in chunks
        ]

    def stats(self) -> dict[str, Any]:
        """Return DB statistics."""
        snap = self._snapshot
        entities = snap.graph_store.get_all_entities()
        relations = snap.graph_store.get_all_relations()
        chunks = snap.doc_store.get_all_chunks()
        hierarchy = snap.doc_store.load_hierarchy()
        return {
            "entities": len(entities),
            "relations": len(relations),
            "chunks": len(chunks),
            "hierarchy_levels": hierarchy.height if hierarchy else 0,
        }

    # ── helpers ──

    @staticmethod
    def _load_corpus(path: str) -> list[dict[str, Any]]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Corpus file not found: {path}")

        documents: list[dict[str, Any]] = []
        text = p.read_text(encoding="utf-8")

        # Try JSONL (one JSON object per line)
        for line in text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                documents.append(doc)
            except json.JSONDecodeError:
                pass

        # If JSONL failed, try as a single JSON array
        if not documents:
            try:
                data = json.loads(text)
                if isinstance(data, list):
                    documents = data
                else:
                    documents = [data]
            except json.JSONDecodeError as exc:
                raise ValueError(f"Cannot parse corpus: {exc}") from exc

        return documents
