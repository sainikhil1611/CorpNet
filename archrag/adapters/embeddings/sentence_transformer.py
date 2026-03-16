"""Embedding adapter: sentence-transformers (default: nomic-embed-text)."""

from __future__ import annotations

from archrag.ports.embedding import EmbeddingPort


class SentenceTransformerEmbedding(EmbeddingPort):
    """Uses the ``sentence-transformers`` library."""

    def __init__(self, model_name: str = "nomic-embed-text-v1.5", device: str = "cpu"):
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name, trust_remote_code=True)
        self._model.to(device)
        self._dim: int = self._model.get_sentence_embedding_dimension()  # type: ignore[assignment]

    def embed(self, text: str) -> list[float]:
        vec = self._model.encode(text, convert_to_numpy=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, convert_to_numpy=True, batch_size=64)
        return vecs.tolist()

    def dimension(self) -> int:
        return self._dim
