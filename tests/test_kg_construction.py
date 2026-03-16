"""Tests for KG construction service."""

from archrag.services.kg_construction import KGConstructionService


class TestKGConstruction:
    def test_build_from_documents(
        self, mock_llm, mock_embedding, in_memory_graph, in_memory_doc
    ):
        svc = KGConstructionService(
            llm=mock_llm,
            embedding=mock_embedding,
            graph_store=in_memory_graph,
            doc_store=in_memory_doc,
            chunk_size=500,
            chunk_overlap=50,
        )

        docs = [
            {
                "title": "Test Doc",
                "context": "Alice works at MIT and collaborates with Bob on AI.",
                "id": 0,
            }
        ]

        kg = svc.build(docs)

        # The mock LLM always returns Alice, Bob, MIT
        assert len(kg.entities) >= 3
        assert len(kg.relations) >= 2

        # Entities should have embeddings
        for entity in kg.entities.values():
            assert entity.embedding is not None
            assert len(entity.embedding) == mock_embedding.dimension()

        # Check persistence
        stored = in_memory_graph.get_all_entities()
        assert len(stored) == len(kg.entities)

    def test_chunking(
        self, mock_llm, mock_embedding, in_memory_graph, in_memory_doc
    ):
        svc = KGConstructionService(
            llm=mock_llm,
            embedding=mock_embedding,
            graph_store=in_memory_graph,
            doc_store=in_memory_doc,
            chunk_size=20,
            chunk_overlap=5,
        )

        docs = [{"title": "Long", "context": "A" * 100, "id": 0}]
        svc.build(docs)

        chunks = in_memory_doc.get_all_chunks()
        # With chunk_size=20 and overlap=5, 100 chars → ceil(100/15) chunks
        assert len(chunks) >= 5
