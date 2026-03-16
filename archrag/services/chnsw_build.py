"""Service: C-HNSW index construction.

Corresponds to ArchRAG §3.1 — C-HNSW Index (Algorithm 3).
Top-down construction: insert nodes from top layer to bottom,
establishing intra-layer links (M nearest neighbours) and
inter-layer links (nearest neighbour in adjacent layer).
"""

from __future__ import annotations

import logging

import numpy as np

from archrag.domain.models import CHNSWIndex, CHNSWNode, CommunityHierarchy
from archrag.ports.document_store import DocumentStorePort
from archrag.ports.embedding import EmbeddingPort
from archrag.ports.graph_store import GraphStorePort
from archrag.ports.vector_index import VectorIndexPort

log = logging.getLogger(__name__)


class CHNSWBuildService:
    """Build the C-HNSW index from a CommunityHierarchy + KG entities."""

    def __init__(
        self,
        embedding: EmbeddingPort,
        vector_index: VectorIndexPort,
        graph_store: GraphStorePort,
        doc_store: DocumentStorePort,
        *,
        M: int = 32,
        ef_construction: int = 100,
    ):
        self._embedding = embedding
        self._vector_index = vector_index
        self._graph_store = graph_store
        self._doc_store = doc_store
        self._M = M
        self._ef_construction = ef_construction

    # ── public ──

    def build(
        self,
        hierarchy: CommunityHierarchy | None = None,
    ) -> CHNSWIndex:
        """Construct the C-HNSW index.

        Layer 0 = entities, layers 1…L = communities from the hierarchy.
        """
        if hierarchy is None:
            hierarchy = self._doc_store.load_hierarchy()
            if hierarchy is None:
                raise RuntimeError("No hierarchy found — run clustering first.")

        index = CHNSWIndex(M=self._M, ef_construction=self._ef_construction)

        # ── Layer 0: entities ──
        entities = self._graph_store.get_all_entities()
        layer0_ids: list[str] = []
        layer0_vecs: list[np.ndarray] = []

        for entity in entities:
            emb = entity.embedding
            if emb is None:
                emb = self._embedding.embed(f"{entity.name}: {entity.description}")
            vec = np.array(emb, dtype=np.float32)
            node = CHNSWNode(
                id=entity.id,
                level=0,
                embedding=emb,
                label=entity.name,
            )
            index.nodes[entity.id] = node
            layer0_ids.append(entity.id)
            layer0_vecs.append(vec)

        if layer0_vecs:
            mat = np.stack(layer0_vecs)
            self._vector_index.add_vectors(layer0_ids, mat, layer=0)

        index.layers.append(layer0_ids)
        log.info("C-HNSW layer 0: %d entity nodes", len(layer0_ids))

        # ── Layers 1…L: communities ──
        for level_idx, level_comms in enumerate(hierarchy.levels):
            layer_num = level_idx + 1
            layer_ids: list[str] = []
            layer_vecs: list[np.ndarray] = []

            for comm in level_comms:
                emb = comm.embedding
                if emb is None:
                    emb = self._embedding.embed(comm.summary)
                vec = np.array(emb, dtype=np.float32)
                node = CHNSWNode(
                    id=comm.id,
                    level=layer_num,
                    embedding=emb,
                    label=comm.summary[:80],
                )
                index.nodes[comm.id] = node
                layer_ids.append(comm.id)
                layer_vecs.append(vec)

            if layer_vecs:
                mat = np.stack(layer_vecs)
                self._vector_index.add_vectors(layer_ids, mat, layer=layer_num)

            index.layers.append(layer_ids)
            log.info("C-HNSW layer %d: %d community nodes", layer_num, len(layer_ids))

        # ── Build intra-layer links ──
        for layer_num, layer_ids in enumerate(index.layers):
            if len(layer_ids) < 2:
                continue
            for nid in layer_ids:
                node = index.nodes[nid]
                query_vec = np.array(node.embedding, dtype=np.float32)
                # Search for M+1 to exclude self
                results = self._vector_index.search(
                    query_vec, min(self._M + 1, len(layer_ids)), layer=layer_num
                )
                neighbours = [rid for rid, _ in results if rid != nid][: self._M]
                node.intra_neighbours = neighbours

        # ── Build inter-layer links ──
        for layer_num in range(1, len(index.layers)):
            upper_ids = index.layers[layer_num]
            for uid in upper_ids:
                upper_node = index.nodes[uid]
                query_vec = np.array(upper_node.embedding, dtype=np.float32)
                # Find nearest in layer below
                results = self._vector_index.search(
                    query_vec, 1, layer=layer_num - 1
                )
                if results:
                    upper_node.inter_link_down = results[0][0]

        # ── Persist the vector index ──
        self._vector_index.save("data/chnsw_vectors.json")
        self._doc_store.put_meta("chnsw_height", str(len(index.layers)))

        log.info(
            "C-HNSW built: %d layers, %d total nodes",
            len(index.layers),
            len(index.nodes),
        )
        return index
