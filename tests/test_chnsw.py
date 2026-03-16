"""Tests for C-HNSW build service."""

import numpy as np

from archrag.domain.models import Community, CommunityHierarchy
from archrag.services.chnsw_build import CHNSWBuildService


class TestCHNSWBuild:
    def test_build(
        self, mock_embedding, numpy_index, in_memory_graph, in_memory_doc, sample_kg
    ):
        # Pre-populate graph store with entities
        for entity in sample_kg.entities.values():
            in_memory_graph.save_entity(entity)

        # Create a simple hierarchy
        c1 = Community(
            id="c1",
            level=0,
            member_ids=["e0", "e1", "e2"],
            summary="Research collaboration",
            embedding=mock_embedding.embed("Research collaboration"),
        )
        c2 = Community(
            id="c2",
            level=0,
            member_ids=["e3", "e4"],
            summary="AI conferences",
            embedding=mock_embedding.embed("AI conferences"),
        )
        hierarchy = CommunityHierarchy(levels=[[c1, c2]])

        svc = CHNSWBuildService(
            embedding=mock_embedding,
            vector_index=numpy_index,
            graph_store=in_memory_graph,
            doc_store=in_memory_doc,
            M=4,
            ef_construction=10,
        )

        index = svc.build(hierarchy)

        # Layer 0 = entities, Layer 1 = communities
        assert index.height == 2
        assert len(index.layers[0]) == 5  # 5 entities
        assert len(index.layers[1]) == 2  # 2 communities

        # All nodes should have embeddings
        for node in index.nodes.values():
            assert node.embedding is not None

        # Intra-layer links should exist for layers with > 1 node
        entity_node = index.nodes["e0"]
        assert len(entity_node.intra_neighbours) > 0

        # Inter-layer links: community nodes should link to entities
        for comm_id in index.layers[1]:
            assert index.nodes[comm_id].inter_link_down is not None
