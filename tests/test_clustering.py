"""Tests for hierarchical clustering service."""

from archrag.services.hierarchical_clustering import HierarchicalClusteringService


class TestHierarchicalClustering:
    def test_build(
        self,
        mock_llm,
        mock_embedding,
        mock_clustering,
        in_memory_graph,
        in_memory_doc,
        sample_kg,
    ):
        svc = HierarchicalClusteringService(
            llm=mock_llm,
            embedding=mock_embedding,
            clustering=mock_clustering,
            graph_store=in_memory_graph,
            doc_store=in_memory_doc,
            max_levels=3,
            similarity_threshold=0.5,
            min_nodes_to_continue=2,
        )

        hierarchy = svc.build(sample_kg)

        assert hierarchy.height >= 1
        assert len(hierarchy.all_communities()) >= 1

        # Each community should have a summary
        for comm in hierarchy.all_communities():
            assert comm.summary
            assert comm.embedding is not None

        # Hierarchy should be persisted
        loaded = in_memory_doc.load_hierarchy()
        assert loaded is not None
        assert loaded.height == hierarchy.height
