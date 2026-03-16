"""Service: Knowledge Graph construction from a text corpus.

Corresponds to ArchRAG §3.1 — KG Construction.
Chunks the corpus, prompts the LLM to extract entities & relations,
merges duplicates, and persists the KG.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from archrag.domain.models import Entity, KnowledgeGraph, Relation, TextChunk
from archrag.ports.document_store import DocumentStorePort
from archrag.ports.embedding import EmbeddingPort
from archrag.ports.graph_store import GraphStorePort
from archrag.ports.llm import LLMPort
from archrag.prompts.extraction import (
    ENTITY_RELATION_EXTRACTION_PROMPT,
    ENTITY_RELATION_EXTRACTION_SYSTEM,
)

log = logging.getLogger(__name__)


class KGConstructionService:
    """Build a knowledge graph from raw text."""

    def __init__(
        self,
        llm: LLMPort,
        embedding: EmbeddingPort,
        graph_store: GraphStorePort,
        doc_store: DocumentStorePort,
        *,
        chunk_size: int = 1200,
        chunk_overlap: int = 100,
    ):
        self._llm = llm
        self._embedding = embedding
        self._graph_store = graph_store
        self._doc_store = doc_store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    # ── public ──

    def build(self, documents: list[dict[str, Any]]) -> KnowledgeGraph:
        """Build a KG from a list of ``{"title": ..., "context": ...}`` dicts.

        Returns the in-memory KG **and** persists it via the graph store port.
        """
        # 1. Chunk
        chunks = self._chunk_documents(documents)
        self._doc_store.save_chunks(chunks)
        log.info("Created %d chunks from %d documents", len(chunks), len(documents))

        # 2. Extract entities & relations per chunk
        kg = KnowledgeGraph()
        name_to_entity: dict[str, Entity] = {}

        for chunk in chunks:
            extracted = self._extract(chunk)
            for ent_data in extracted.get("entities", []):
                name = ent_data.get("name", "").strip()
                if not name:
                    continue
                key = name.lower()
                if key in name_to_entity:
                    # Merge descriptions
                    existing = name_to_entity[key]
                    if ent_data.get("description", ""):
                        existing.description += " " + ent_data["description"]
                    existing.source_chunk_ids.append(chunk.id)
                else:
                    entity = Entity(
                        name=name,
                        description=ent_data.get("description", ""),
                        entity_type=ent_data.get("type", ""),
                        source_chunk_ids=[chunk.id],
                    )
                    name_to_entity[key] = entity
                    kg.add_entity(entity)

            for rel_data in extracted.get("relations", []):
                src_name = rel_data.get("source", "").strip().lower()
                tgt_name = rel_data.get("target", "").strip().lower()
                if src_name in name_to_entity and tgt_name in name_to_entity:
                    rel = Relation(
                        source_id=name_to_entity[src_name].id,
                        target_id=name_to_entity[tgt_name].id,
                        description=rel_data.get("description", ""),
                        source_chunk_ids=[chunk.id],
                    )
                    kg.add_relation(rel)

        # 3. Compute entity embeddings
        entity_list = list(kg.entities.values())
        if entity_list:
            texts = [f"{e.name}: {e.description}" for e in entity_list]
            embeddings = self._embedding.embed_batch(texts)
            for entity, emb in zip(entity_list, embeddings):
                entity.embedding = emb

        # 4. Persist
        self._graph_store.save_entities(entity_list)
        self._graph_store.save_relations(kg.relations)

        log.info(
            "KG built: %d entities, %d relations",
            len(kg.entities),
            len(kg.relations),
        )
        return kg

    # ── private helpers ──

    def _chunk_documents(self, documents: list[dict[str, Any]]) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        for doc in documents:
            text = doc.get("context", doc.get("text", ""))
            title = doc.get("title", "")
            start = 0
            while start < len(text):
                end = start + self._chunk_size
                chunk_text = text[start:end]
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        source_doc=title,
                        metadata={"doc_id": doc.get("id", "")},
                    )
                )
                start += self._chunk_size - self._chunk_overlap
        return chunks

    def _extract(self, chunk: TextChunk) -> dict[str, Any]:
        prompt = ENTITY_RELATION_EXTRACTION_PROMPT.format(text=chunk.text)
        try:
            result = self._llm.generate_json(
                prompt, system=ENTITY_RELATION_EXTRACTION_SYSTEM
            )
            return result
        except (json.JSONDecodeError, Exception) as exc:
            log.warning("Extraction failed for chunk %s: %s", chunk.id, exc)
            return {"entities": [], "relations": []}
