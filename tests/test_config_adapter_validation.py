"""Tests for clean adapter/env validation in archrag.config.

Bug: selecting `adapter: openai` in config.yaml with the `openai` extra
not installed, or with OPENAI_API_KEY unset, used to surface a raw
traceback (ModuleNotFoundError / openai.OpenAIError) from deep inside
build_embedding()/build_llm(). These tests prove both failure modes now
raise a clean AdapterConfigError with an actionable message instead.
"""

from __future__ import annotations

import sys

import pytest

from archrag.config import AdapterConfigError, build_embedding, build_llm


def test_build_embedding_openai_missing_package_raises_clean_error(monkeypatch):
    # Simulate the 'openai' package not being installed: sys.modules[name] = None
    # forces `import openai` to raise ImportError.
    monkeypatch.setitem(sys.modules, "openai", None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-doesnt-matter")

    with pytest.raises(AdapterConfigError, match="openai.*package.*isn't installed"):
        build_embedding({"adapter": "openai"})


def test_build_embedding_openai_missing_api_key_raises_clean_error(monkeypatch):
    # Simulate the 'openai' package being installed (fake module) but no key set.
    import types

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = lambda **kwargs: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(AdapterConfigError, match="OPENAI_API_KEY is not set"):
        build_embedding({"adapter": "openai"})


def test_build_llm_openai_missing_package_raises_clean_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-doesnt-matter")

    with pytest.raises(AdapterConfigError, match="openai.*package.*isn't installed"):
        build_llm({"adapter": "openai"})


def test_build_llm_openai_missing_api_key_raises_clean_error(monkeypatch):
    import types

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = lambda **kwargs: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(AdapterConfigError, match="OPENAI_API_KEY is not set"):
        build_llm({"adapter": "openai"})


def test_build_embedding_openai_succeeds_when_package_and_key_present(monkeypatch):
    import types

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = lambda **kwargs: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

    embedding = build_embedding({"adapter": "openai", "dimension": 1536})
    assert embedding.dimension() == 1536


def test_build_llm_openai_succeeds_when_package_and_key_present(monkeypatch):
    import types

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = lambda **kwargs: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

    llm = build_llm({"adapter": "openai"})
    assert llm is not None


def test_non_openai_adapters_are_unaffected(monkeypatch):
    """The validation is scoped to the openai adapter only: the
    sentence_transformer embedding adapter must not raise AdapterConfigError
    just because OPENAI_API_KEY is unset."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # sentence-transformers isn't installed in this environment (and is
    # heavy even when it is), so inject a fake module — same pattern used
    # above for the 'openai' package — to keep this test fast and offline.
    import types

    fake_st_module = types.ModuleType("sentence_transformers")

    class _FakeSentenceTransformer:
        def __init__(self, model_name: str, trust_remote_code: bool = False):
            self._model_name = model_name

        def to(self, device: str):
            return self

        def get_sentence_embedding_dimension(self) -> int:
            return 384

    fake_st_module.SentenceTransformer = _FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st_module)

    embedding = build_embedding({"adapter": "sentence_transformer"})
    assert embedding.dimension() == 384
