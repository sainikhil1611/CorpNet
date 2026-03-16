"""Shared test fixtures — mock ports and sample data."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from archrag.adapters.indexes.numpy_vector import NumpyVectorIndex
from archrag.adapters.stores.in_memory_document import InMemoryDocumentStore
from archrag.adapters.stores.in_memory_graph import InMemoryGraphStore
from archrag.domain.models import (
    Community,
    CommunityHierarchy,
    Entity,
    KnowledgeGraph,
    Relation,
)
from archrag.ports.clustering import ClusteringPort, WeightedEdge
from archrag.ports.embedding import EmbeddingPort
from archrag.ports.llm import LLMPort


# ── Mock Embedding ──


class MockEmbedding(EmbeddingPort):
    """Returns deterministic embeddings based on hash of text."""

    DIM = 8

    def embed(self, text: str) -> list[float]:
        rng = np.random.RandomState(hash(text) % 2**31)
        return rng.randn(self.DIM).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def dimension(self) -> int:
        return self.DIM


# ── Mock LLM ──


class MockLLM(LLMPort):
    """Returns canned responses for testing."""

    def generate(self, prompt: str, *, system: str = "") -> str:
        if "summarise" in prompt.lower() or "summary" in prompt.lower():
            return "This community discusses key entities and their relationships."
        if "merge" in prompt.lower() or "analyst" in prompt.lower():
            return "Based on the retrieved information, the answer is: test answer."
        return "Mock LLM response."

    def generate_json(self, prompt: str, *, system: str = "") -> dict[str, Any]:
        if "extract" in prompt.lower() or "entities" in prompt.lower():
            return {
                "entities": [
                    {"name": "Alice", "type": "PERSON", "description": "A researcher"},
                    {"name": "Bob", "type": "PERSON", "description": "A professor"},
                    {"name": "MIT", "type": "ORGANIZATION", "description": "A university"},
                ],
                "relations": [
                    {"source": "Alice", "target": "Bob", "description": "collaborates with"},
                    {"source": "Alice", "target": "MIT", "description": "works at"},
                ],
            }
        if "filter" in prompt.lower() or "points" in prompt.lower():
            return {
                "points": [
                    {"description": "Key point about the topic.", "score": 85},
                    {"description": "Supporting detail.", "score": 60},
                ]
            }
        return {"result": "mock"}


# ── Mock Clustering ──


class MockClustering(ClusteringPort):
    """Splits nodes into groups of ~3."""

    def cluster(
        self,
        node_ids: list[str],
        edges: list[WeightedEdge],
    ) -> list[list[str]]:
        groups: list[list[str]] = []
        current: list[str] = []
        for nid in node_ids:
            current.append(nid)
            if len(current) >= 3:
                groups.append(current)
                current = []
        if current:
            groups.append(current)
        return groups


# ── Fixtures ──


@pytest.fixture
def mock_embedding():
    return MockEmbedding()


@pytest.fixture
def mock_llm():
    return MockLLM()


@pytest.fixture
def mock_clustering():
    return MockClustering()


@pytest.fixture
def in_memory_graph():
    return InMemoryGraphStore()


@pytest.fixture
def in_memory_doc():
    return InMemoryDocumentStore()


@pytest.fixture
def numpy_index():
    return NumpyVectorIndex()


@pytest.fixture
def sample_kg():
    """A small KG with 5 entities and 4 relations."""
    kg = KnowledgeGraph()
    rng = np.random.RandomState(42)
    names = ["Alice", "Bob", "MIT", "AI Research", "NeurIPS"]
    descs = [
        "A researcher in AI",
        "A professor of CS",
        "A university in Cambridge",
        "Research field of artificial intelligence",
        "A major ML conference",
    ]
    entities = []
    for i, (name, desc) in enumerate(zip(names, descs)):
        e = Entity(
            id=f"e{i}",
            name=name,
            description=desc,
            embedding=rng.randn(8).tolist(),
        )
        kg.add_entity(e)
        entities.append(e)

    rels = [
        ("e0", "e1", "collaborates with"),
        ("e0", "e2", "works at"),
        ("e1", "e2", "affiliated with"),
        ("e0", "e3", "researches"),
    ]
    for src, tgt, desc in rels:
        kg.add_relation(Relation(source_id=src, target_id=tgt, description=desc))

    return kg
