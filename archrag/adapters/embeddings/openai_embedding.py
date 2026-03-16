"""Embedding adapter: OpenAI API."""

from __future__ import annotations

from archrag.ports.embedding import EmbeddingPort


class OpenAIEmbedding(EmbeddingPort):
    """Uses the OpenAI embeddings endpoint."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        dimension: int = 1536,
    ):
        from openai import OpenAI  # lazy

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._dim = dimension

    def embed(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(input=[text], model=self._model)
        return resp.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(input=texts, model=self._model)
        return [d.embedding for d in resp.data]

    def dimension(self) -> int:
        return self._dim
