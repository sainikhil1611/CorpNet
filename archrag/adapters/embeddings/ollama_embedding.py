"""Embedding adapter: Ollama local server."""

from __future__ import annotations

from archrag.ports.embedding import EmbeddingPort


class OllamaEmbedding(EmbeddingPort):
    """Calls the Ollama /api/embed endpoint."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        dimension: int = 768,
    ):
        from ollama import Client  # lazy

        self._client = Client(host=base_url)
        self._model = model
        self._dim = dimension

    def embed(self, text: str) -> list[float]:
        resp = self._client.embed(model=self._model, input=text)
        return list(resp["embeddings"][0])

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embed(model=self._model, input=texts)
        return [list(e) for e in resp["embeddings"]]

    def dimension(self) -> int:
        return self._dim
