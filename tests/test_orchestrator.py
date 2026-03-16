"""Integration test: full pipeline with in-memory adapters."""

import json
import tempfile
from pathlib import Path

from archrag.services.orchestrator import ArchRAGOrchestrator
from tests.conftest import MockClustering, MockEmbedding, MockLLM


class TestOrchestrator:
    def test_full_pipeline(self, in_memory_graph, in_memory_doc, numpy_index):
        orch = ArchRAGOrchestrator(
            llm=MockLLM(),
            embedding=MockEmbedding(),
            graph_store=in_memory_graph,
            doc_store=in_memory_doc,
            vector_index=numpy_index,
            clustering=MockClustering(),
            chunk_size=500,
            chunk_overlap=50,
            max_levels=2,
            M=4,
            ef_construction=10,
            k_per_layer=3,
        )

        # Create a temp corpus file
        corpus = [
            {
                "title": "AI Research",
                "context": "Alice is a researcher at MIT who works on artificial intelligence with Bob.",
                "id": 0,
            },
            {
                "title": "Conferences",
                "context": "NeurIPS is a major machine learning conference attended by researchers worldwide.",
                "id": 1,
            },
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            for doc in corpus:
                f.write(json.dumps(doc) + "\n")
            corpus_path = f.name

        try:
            # Offline indexing
            orch.index(corpus_path)

            # Online query
            answer = orch.query("Who does Alice collaborate with?")

            assert isinstance(answer, str)
            assert len(answer) > 0
        finally:
            Path(corpus_path).unlink(missing_ok=True)
