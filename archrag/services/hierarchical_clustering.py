"""Service: LLM-based Hierarchical Clustering.

Corresponds to ArchRAG §3.1 — Algorithm 1.
Iteratively: augment graph (KNN), weight edges, cluster, summarise,
build upper-level graph, repeat.  Produces a CommunityHierarchy.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from archrag.domain.models import (
    Community,
    CommunityHierarchy,
    Entity,
    KnowledgeGraph,
    Relation,
)
from archrag.ports.clustering import ClusteringPort, WeightedEdge
from archrag.ports.document_store import DocumentStorePort
from archrag.ports.embedding import EmbeddingPort
from archrag.ports.graph_store import GraphStorePort
from archrag.ports.llm import LLMPort
from archrag.prompts.summarization import (
    COMMUNITY_SUMMARY_PROMPT,
    COMMUNITY_SUMMARY_SYSTEM,
)

log = logging.getLogger(__name__)


class HierarchicalClusteringService:
    """Detect attributed communities and organise them hierarchically."""

    def __init__(
        self,
        llm: LLMPort,
        embedding: EmbeddingPort,
        clustering: ClusteringPort,
        graph_store: GraphStorePort,
        doc_store: DocumentStorePort,
        *,
        max_levels: int = 5,
        similarity_threshold: float = 0.7,
        min_nodes_to_continue: int = 3,
    ):
        self._llm = llm
        self._embedding = embedding
        self._clustering = clustering
        self._graph_store = graph_store
        self._doc_store = doc_store
        self._max_levels = max_levels
        self._sim_threshold = similarity_threshold
        self._min_nodes = min_nodes_to_continue

    # ── public ──

    def build(self, kg: KnowledgeGraph | None = None) -> CommunityHierarchy:
        """Run the iterative hierarchical clustering (Algorithm 1)."""
        if kg is None:
            kg = self._load_kg()

        hierarchy = CommunityHierarchy()

        # Current-level "graph" represented as nodes + edges
        current_nodes: dict[str, _NodeInfo] = {}
        for entity in kg.entities.values():
            emb = entity.embedding or self._embedding.embed(
                f"{entity.name}: {entity.description}"
            )
            current_nodes[entity.id] = _NodeInfo(
                id=entity.id,
                text=f"{entity.name}: {entity.description}",
                embedding=np.array(emb, dtype=np.float32),
            )

        current_edges: list[tuple[str, str]] = [
            (r.source_id, r.target_id) for r in kg.relations
        ]

        for level in range(self._max_levels):
            if len(current_nodes) < self._min_nodes:
                log.info("Stopping at level %d: too few nodes (%d)", level, len(current_nodes))
                break

            log.info(
                "Clustering level %d: %d nodes, %d edges",
                level,
                len(current_nodes),
                len(current_edges),
            )

            # Step 1: Augment graph — add KNN edges by attribute similarity
            augmented_edges = self._augment_knn(current_nodes, current_edges)

            # Step 2: Compute edge weights as 1 - cosine(u, v)
            weighted = self._compute_weights(current_nodes, augmented_edges)

            # Step 3: Cluster
            node_ids = list(current_nodes.keys())
            communities_ids = self._clustering.cluster(node_ids, weighted)

            if len(communities_ids) <= 1 and level > 0:
                log.info("Single community at level %d — stopping", level)
                break

            # Step 4: Summarise each community via LLM
            level_communities: list[Community] = []
            for comm_member_ids in communities_ids:
                member_texts = [current_nodes[mid].text for mid in comm_member_ids if mid in current_nodes]
                summary = self._summarise(member_texts)
                emb = self._embedding.embed(summary)
                community = Community(
                    level=level,
                    member_ids=comm_member_ids,
                    summary=summary,
                    embedding=emb,
                )
                level_communities.append(community)

            hierarchy.levels.append(level_communities)

            # Step 5: Build upper-level graph — each community is a node
            upper_nodes: dict[str, _NodeInfo] = {}
            for comm in level_communities:
                upper_nodes[comm.id] = _NodeInfo(
                    id=comm.id,
                    text=comm.summary,
                    embedding=np.array(comm.embedding, dtype=np.float32),
                )

            # Connect communities whose members share edges
            member_to_comm: dict[str, str] = {}
            for comm in level_communities:
                for mid in comm.member_ids:
                    member_to_comm[mid] = comm.id

            upper_edges_set: set[tuple[str, str]] = set()
            for src, tgt in augmented_edges:
                c1 = member_to_comm.get(src)
                c2 = member_to_comm.get(tgt)
                if c1 and c2 and c1 != c2:
                    pair = (min(c1, c2), max(c1, c2))
                    upper_edges_set.add(pair)

            current_nodes = upper_nodes
            current_edges = list(upper_edges_set)

        # Persist
        self._doc_store.save_hierarchy(hierarchy)
        log.info(
            "Hierarchy built: %d levels, %d total communities",
            hierarchy.height,
            len(hierarchy.all_communities()),
        )
        return hierarchy

    # ── private helpers ──

    def _load_kg(self) -> KnowledgeGraph:
        entities = self._graph_store.get_all_entities()
        relations = self._graph_store.get_all_relations()
        kg = KnowledgeGraph()
        for e in entities:
            kg.add_entity(e)
        kg.relations = relations
        return kg

    def _augment_knn(
        self,
        nodes: dict[str, _NodeInfo],
        existing_edges: list[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        """Add edges between nodes whose cosine similarity > threshold."""
        node_ids = list(nodes.keys())
        if len(node_ids) < 2:
            return list(existing_edges)

        mat = np.stack([nodes[nid].embedding for nid in node_ids])
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10
        mat_normed = mat / norms
        sim_matrix = mat_normed @ mat_normed.T

        existing_set = set(existing_edges) | {(b, a) for a, b in existing_edges}
        augmented = list(existing_edges)

        # Compute average degree for K
        avg_degree = max(1, (2 * len(existing_edges)) // max(len(node_ids), 1))
        k = max(avg_degree, 2)

        for i, nid in enumerate(node_ids):
            sims = sim_matrix[i]
            # Get top-k neighbours (excluding self)
            top_indices = np.argsort(-sims)
            added = 0
            for j in top_indices:
                if added >= k:
                    break
                if i == j:
                    continue
                other = node_ids[j]
                if sims[j] >= self._sim_threshold:
                    pair = (nid, other)
                    if pair not in existing_set and (other, nid) not in existing_set:
                        augmented.append(pair)
                        existing_set.add(pair)
                        added += 1

        return augmented

    def _compute_weights(
        self,
        nodes: dict[str, _NodeInfo],
        edges: list[tuple[str, str]],
    ) -> list[WeightedEdge]:
        weighted: list[WeightedEdge] = []
        for src, tgt in edges:
            if src in nodes and tgt in nodes:
                emb_s = nodes[src].embedding
                emb_t = nodes[tgt].embedding
                cos_sim = float(
                    np.dot(emb_s, emb_t)
                    / (np.linalg.norm(emb_s) * np.linalg.norm(emb_t) + 1e-10)
                )
                # Weight = similarity (higher = closer), used by Leiden
                weighted.append(WeightedEdge(source=src, target=tgt, weight=max(cos_sim, 0.0)))
        return weighted

    def _summarise(self, member_texts: list[str]) -> str:
        entities_block = "\n".join(f"- {t}" for t in member_texts[:30])
        prompt = COMMUNITY_SUMMARY_PROMPT.format(
            entities_text=entities_block,
            relations_text="(see entity descriptions)",
        )
        try:
            return self._llm.generate(prompt, system=COMMUNITY_SUMMARY_SYSTEM)
        except Exception as exc:
            log.warning("Summarisation failed: %s", exc)
            return " | ".join(member_texts[:5])


# ── internal helper ──

class _NodeInfo:
    """Lightweight carrier for node data during clustering."""

    __slots__ = ("id", "text", "embedding")

    def __init__(self, id: str, text: str, embedding: Any):
        self.id = id
        self.text = text
        self.embedding = embedding
