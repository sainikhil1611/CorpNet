"""Clustering adapter: Leiden algorithm (via leidenalg + igraph)."""

from __future__ import annotations

from archrag.ports.clustering import ClusteringPort, WeightedEdge


class LeidenClustering(ClusteringPort):
    """Weighted Leiden community detection."""

    def __init__(self, resolution: float = 1.0):
        self._resolution = resolution

    def cluster(
        self,
        node_ids: list[str],
        edges: list[WeightedEdge],
    ) -> list[list[str]]:
        import igraph as ig  # lazy
        import leidenalg  # lazy

        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        g = ig.Graph(n=len(node_ids), directed=False)
        edge_tuples = []
        weights = []
        for e in edges:
            s = id_to_idx.get(e.source)
            t = id_to_idx.get(e.target)
            if s is not None and t is not None and s != t:
                edge_tuples.append((s, t))
                weights.append(e.weight)

        if edge_tuples:
            g.add_edges(edge_tuples)
            g.es["weight"] = weights

        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight" if weights else None,
            resolution_parameter=self._resolution,
        )

        communities: list[list[str]] = []
        for community_indices in partition:
            comm = [node_ids[i] for i in community_indices]
            if comm:
                communities.append(comm)

        # If no communities found, put every node in its own community
        if not communities:
            communities = [[nid] for nid in node_ids]

        return communities
