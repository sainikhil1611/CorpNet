"""Configuration loading and adapter factory.

Reads a YAML config file and instantiates the correct adapter
for each port, then wires them into the ArchRAGOrchestrator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Walk up from this file (archrag/config.py) to the project root and load .env
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)

from archrag.ports.clustering import ClusteringPort
from archrag.ports.document_store import DocumentStorePort
from archrag.ports.embedding import EmbeddingPort
from archrag.ports.graph_store import GraphStorePort
from archrag.ports.llm import LLMPort
from archrag.ports.vector_index import VectorIndexPort
from archrag.services.orchestrator import ArchRAGOrchestrator


def load_config(path: str = "config.yaml") -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(p) as f:
        return yaml.safe_load(f)


# ── Adapter factories ──


def build_embedding(cfg: dict[str, Any]) -> EmbeddingPort:
    adapter = cfg.get("adapter", "sentence_transformer")
    model = cfg.get("model", "nomic-embed-text-v1.5")
    dimension = cfg.get("dimension", 768)

    if adapter == "sentence_transformer":
        from archrag.adapters.embeddings.sentence_transformer import (
            SentenceTransformerEmbedding,
        )
        return SentenceTransformerEmbedding(model_name=model)

    elif adapter == "openai":
        from archrag.adapters.embeddings.openai_embedding import OpenAIEmbedding
        return OpenAIEmbedding(model=model, dimension=dimension)

    elif adapter == "ollama":
        from archrag.adapters.embeddings.ollama_embedding import OllamaEmbedding
        base_url = cfg.get("base_url", "http://localhost:11434")
        return OllamaEmbedding(model=model, base_url=base_url, dimension=dimension)

    raise ValueError(f"Unknown embedding adapter: {adapter}")


def build_llm(cfg: dict[str, Any]) -> LLMPort:
    adapter = cfg.get("adapter", "ollama")
    model = cfg.get("model", "llama3.1:8b")
    temperature = cfg.get("temperature", 0.0)

    if adapter == "ollama":
        from archrag.adapters.llms.ollama_llm import OllamaLLM
        base_url = cfg.get("base_url", "http://localhost:11434")
        return OllamaLLM(model=model, base_url=base_url, temperature=temperature)

    elif adapter == "openai":
        from archrag.adapters.llms.openai_llm import OpenAILLM
        return OpenAILLM(model=model, temperature=temperature)

    raise ValueError(f"Unknown LLM adapter: {adapter}")


def build_graph_store(cfg: dict[str, Any]) -> GraphStorePort:
    adapter = cfg.get("adapter", "sqlite")
    db_path = cfg.get("path", "data/archrag.db")

    if adapter == "sqlite":
        from archrag.adapters.stores.sqlite_graph import SQLiteGraphStore
        return SQLiteGraphStore(db_path=db_path)

    elif adapter == "in_memory":
        from archrag.adapters.stores.in_memory_graph import InMemoryGraphStore
        return InMemoryGraphStore()

    raise ValueError(f"Unknown graph_store adapter: {adapter}")


def build_document_store(cfg: dict[str, Any]) -> DocumentStorePort:
    adapter = cfg.get("adapter", "sqlite")
    db_path = cfg.get("path", "data/archrag.db")

    if adapter == "sqlite":
        from archrag.adapters.stores.sqlite_document import SQLiteDocumentStore
        return SQLiteDocumentStore(db_path=db_path)

    elif adapter == "in_memory":
        from archrag.adapters.stores.in_memory_document import InMemoryDocumentStore
        return InMemoryDocumentStore()

    raise ValueError(f"Unknown document_store adapter: {adapter}")


def build_vector_index(cfg: dict[str, Any]) -> VectorIndexPort:
    adapter = cfg.get("adapter", "numpy")

    if adapter == "numpy":
        from archrag.adapters.indexes.numpy_vector import NumpyVectorIndex
        return NumpyVectorIndex()

    raise ValueError(f"Unknown vector_index adapter: {adapter}")


def build_clustering(cfg: dict[str, Any]) -> ClusteringPort:
    adapter = cfg.get("adapter", "leiden")
    resolution = cfg.get("resolution", 1.0)

    if adapter == "leiden":
        from archrag.adapters.clustering.leiden import LeidenClustering
        return LeidenClustering(resolution=resolution)

    raise ValueError(f"Unknown clustering adapter: {adapter}")


# ── Top-level builder ──


def build_orchestrator(config_path: str = "config.yaml") -> ArchRAGOrchestrator:
    """Load config and wire all adapters into the orchestrator."""
    import logging
    log = logging.getLogger(__name__)

    cfg = load_config(config_path)

    log.info("  → building embedding adapter …")
    embedding = build_embedding(cfg.get("embedding", {}))
    log.info("  ✓ embedding ready")

    log.info("  → building LLM adapter …")
    llm = build_llm(cfg.get("llm", {}))
    log.info("  ✓ LLM ready")

    log.info("  → building graph store …")
    graph_store = build_graph_store(cfg.get("graph_store", {}))
    log.info("  ✓ graph store ready")

    log.info("  → building document store …")
    doc_store = build_document_store(cfg.get("document_store", {}))
    log.info("  ✓ document store ready")

    log.info("  → building vector index …")
    vector_index = build_vector_index(cfg.get("vector_index", {}))
    log.info("  ✓ vector index ready")

    log.info("  → building clustering …")
    clustering = build_clustering(cfg.get("clustering", {}))
    log.info("  ✓ clustering ready")

    indexing_cfg = cfg.get("indexing", {})
    retrieval_cfg = cfg.get("retrieval", {})
    chnsw_cfg = cfg.get("chnsw", {})

    return ArchRAGOrchestrator(
        llm=llm,
        embedding=embedding,
        graph_store=graph_store,
        doc_store=doc_store,
        vector_index=vector_index,
        clustering=clustering,
        chunk_size=indexing_cfg.get("chunk_size", 1200),
        chunk_overlap=indexing_cfg.get("chunk_overlap", 100),
        max_levels=indexing_cfg.get("max_hierarchy_levels", 5),
        similarity_threshold=indexing_cfg.get("similarity_threshold", 0.7),
        M=chnsw_cfg.get("M", 32),
        ef_construction=chnsw_cfg.get("ef_construction", 100),
        k_per_layer=retrieval_cfg.get("k_per_layer", 5),
    )
