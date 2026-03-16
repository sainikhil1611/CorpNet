"""Service: Hierarchical Search on the C-HNSW index.

Corresponds to ArchRAG §3.2 — Algorithm 2.
Start from top layer, SearchLayer per layer, follow inter-layer
links downward, collect k nearest neighbours at each level.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from archrag.domain.models import CHNSWIndex, CHNSWNode, SearchResult
from archrag.ports.document_store import DocumentStorePort
from archrag.ports.embedding import EmbeddingPort
from archrag.ports.graph_store import GraphStorePort
from archrag.ports.vector_index import VectorIndexPort

log = logging.getLogger(__name__)


class HierarchicalSearchService:
    """Perform hierarchical search on the C-HNSW index."""

    def __init__(
        self,
        embedding: EmbeddingPort,
        vector_index: VectorIndexPort,
        graph_store: GraphStorePort,
        doc_store: DocumentStorePort,
        *,
        k_per_layer: int = 5,
    ):
        self._embedding = embedding
        self._vector_index = vector_index
        self._graph_store = graph_store
        self._doc_store = doc_store
        self._k = k_per_layer

    # ── public ──

    def search(
        self,
        query: str,
        index: CHNSWIndex | None = None,
    ) -> list[list[SearchResult]]:
        """Return search results grouped by layer (index 0 = entities).

        Implements Algorithm 2: hierarchical search with reuse of
        intermediate results across layers.
        """
        # Embed query
        q_vec = np.array(self._embedding.embed(query), dtype=np.float32)

        # Determine number of layers
        if index is not None:
            num_layers = index.height
        else:
            height_str = self._doc_store.get_meta("chnsw_height")
            num_layers = int(height_str) if height_str else 1

        all_results: list[list[SearchResult]] = []

        # Start from the top layer and search downward
        entry_candidates: list[str] | None = None

        for layer in range(num_layers - 1, -1, -1):
            # SearchLayer: find k nearest neighbours at this layer
            results = self._vector_index.search(
                q_vec,
                self._k,
                layer=layer,
                candidate_ids=entry_candidates,
            )

            layer_results: list[SearchResult] = []
            for node_id, dist in results:
                text = self._get_text_for_node(node_id, layer, index)
                layer_results.append(
                    SearchResult(
                        node_id=node_id,
                        level=layer,
                        distance=dist,
                        text=text,
                    )
                )

            all_results.append(layer_results)

            # Follow inter-layer link: use the closest result's
            # inter-layer link as the starting point for the next layer.
            # For the vector index approach, we simply let the next layer
            # search broadly (no candidate restriction).
            entry_candidates = None  # search full layer below

        # Reverse so index 0 = layer 0 (entities)
        all_results.reverse()

        # For layer 0 results, also fetch relationship context
        if all_results and all_results[0]:
            self._enrich_entity_results(all_results[0])

        return all_results

    # ── private helpers ──

    def _get_text_for_node(
        self,
        node_id: str,
        layer: int,
        index: CHNSWIndex | None,
    ) -> str:
        if layer == 0:
            # Entity
            entity = self._graph_store.get_entity(node_id)
            if entity:
                return f"{entity.name}: {entity.description}"
            return ""
        else:
            # Community
            comm = self._doc_store.get_community(node_id)
            if comm:
                return comm.summary
            # Fall back to index label
            if index and node_id in index.nodes:
                return index.nodes[node_id].label
            return ""

    def _enrich_entity_results(self, entity_results: list[SearchResult]) -> None:
        """Add relationship info for retrieved entities (textual subgraph)."""
        entity_ids = {r.node_id for r in entity_results}
        for result in entity_results:
            rels = self._graph_store.get_relations_for(result.node_id)
            rel_texts: list[str] = []
            for r in rels:
                if r.source_id in entity_ids or r.target_id in entity_ids:
                    src = self._graph_store.get_entity(r.source_id)
                    tgt = self._graph_store.get_entity(r.target_id)
                    src_name = src.name if src else r.source_id
                    tgt_name = tgt.name if tgt else r.target_id
                    rel_texts.append(f"{src_name} -> {tgt_name}: {r.description}")
            if rel_texts:
                result.text += "\nRelationships:\n" + "\n".join(rel_texts)
