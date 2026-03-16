"""Tests for hierarchical search service."""

import numpy as np

from archrag.domain.models import Community, CommunityHierarchy, Entity
from archrag.services.chnsw_build import CHNSWBuildService
from archrag.services.hierarchical_search import HierarchicalSearchService


class TestHierarchicalSearch:
    def test_search(
        self, mock_embedding, numpy_index, in_memory_graph, in_memory_doc, sample_kg
    ):
        # Setup: populate stores
        for entity in sample_kg.entities.values():
            in_memory_graph.save_entity(entity)
        for rel in sample_kg.relations:
            in_memory_graph.save_relation(rel)

        c1 = Community(
            id="c1",
            level=0,
            member_ids=["e0", "e1", "e2"],
            summary="Research collaboration",
            embedding=mock_embedding.embed("Research collaboration"),
        )
        hierarchy = CommunityHierarchy(levels=[[c1]])
        in_memory_doc.save_hierarchy(hierarchy)

        # Build index
        build_svc = CHNSWBuildService(
            embedding=mock_embedding,
            vector_index=numpy_index,
            graph_store=in_memory_graph,
            doc_store=in_memory_doc,
            M=4,
            ef_construction=10,
        )
        index = build_svc.build(hierarchy)

        # Search
        search_svc = HierarchicalSearchService(
            embedding=mock_embedding,
            vector_index=numpy_index,
            graph_store=in_memory_graph,
            doc_store=in_memory_doc,
            k_per_layer=3,
        )

        results = search_svc.search("Who collaborates with Alice?", index)

        # Should have results for each layer
        assert len(results) == 2  # layer 0 (entities) + layer 1 (communities)
        assert len(results[0]) > 0  # at least some entity results
        assert len(results[1]) > 0  # at least some community results

        # Entity results should have text
        for r in results[0]:
            assert r.text
