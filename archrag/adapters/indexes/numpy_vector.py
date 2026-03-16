"""Vector index adapter: pure NumPy (no FAISS dependency)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from archrag.ports.vector_index import VectorIndexPort


class NumpyVectorIndex(VectorIndexPort):
    """Brute-force + sorted cosine-distance index backed by NumPy arrays.

    Organised into layers to mirror the C-HNSW structure.
    Suitable for small–medium corpora (< 1M vectors).
    """

    def __init__(self) -> None:
        # layer → {id: vector}
        self._layers: dict[int, dict[str, np.ndarray]] = {}
        # flat lookup across all layers
        self._all_vectors: dict[str, np.ndarray] = {}

    # ── write ──

    def add_vectors(
        self,
        ids: list[str],
        vectors: np.ndarray,
        *,
        layer: int = 0,
    ) -> None:
        if layer not in self._layers:
            self._layers[layer] = {}
        for i, vid in enumerate(ids):
            vec = vectors[i].astype(np.float32)
            self._layers[layer][vid] = vec
            self._all_vectors[vid] = vec

    # ── read ──

    def search(
        self,
        query: np.ndarray,
        k: int,
        *,
        layer: int = 0,
        candidate_ids: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        pool = self._layers.get(layer, {})
        if not pool:
            return []

        if candidate_ids is not None:
            pool = {vid: pool[vid] for vid in candidate_ids if vid in pool}
        if not pool:
            return []

        ids_list = list(pool.keys())
        mat = np.stack([pool[vid] for vid in ids_list])  # (N, d)
        q = query.astype(np.float32).reshape(1, -1)

        # Cosine distance = 1 - cosine_similarity
        q_norm = q / (np.linalg.norm(q) + 1e-10)
        mat_norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10)
        sims = (mat_norm @ q_norm.T).flatten()
        dists = 1.0 - sims

        top_k_idx = np.argsort(dists)[:k]
        return [(ids_list[i], float(dists[i])) for i in top_k_idx]

    def get_vector(self, id: str) -> np.ndarray | None:
        return self._all_vectors.get(id)

    # ── persistence ──

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data: dict = {}
        for layer, vecs in self._layers.items():
            data[str(layer)] = {vid: vec.tolist() for vid, vec in vecs.items()}
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self._layers.clear()
        self._all_vectors.clear()
        for layer_str, vecs in data.items():
            layer = int(layer_str)
            self._layers[layer] = {}
            for vid, vec_list in vecs.items():
                arr = np.array(vec_list, dtype=np.float32)
                self._layers[layer][vid] = arr
                self._all_vectors[vid] = arr

    def clear(self) -> None:
        self._layers.clear()
        self._all_vectors.clear()

    def clone(self) -> "NumpyVectorIndex":
        """Create an independent deep copy of the index."""
        import copy
        new = NumpyVectorIndex()
        new._layers = {
            layer: {vid: vec.copy() for vid, vec in vecs.items()}
            for layer, vecs in self._layers.items()
        }
        new._all_vectors = {vid: vec.copy() for vid, vec in self._all_vectors.items()}
        return new
