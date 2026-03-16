"""Tests for adaptive filtering service."""

from archrag.domain.models import SearchResult
from archrag.services.adaptive_filtering import AdaptiveFilteringService


class TestAdaptiveFiltering:
    def test_generate_answer(self, mock_llm):
        svc = AdaptiveFilteringService(llm=mock_llm)

        results = [
            [
                SearchResult(node_id="e0", level=0, distance=0.1, text="Alice: researcher"),
                SearchResult(node_id="e1", level=0, distance=0.2, text="Bob: professor"),
            ],
            [
                SearchResult(
                    node_id="c1",
                    level=1,
                    distance=0.15,
                    text="Community about AI research",
                ),
            ],
        ]

        answer = svc.generate_answer("Who is Alice?", results)

        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_empty_results(self, mock_llm):
        svc = AdaptiveFilteringService(llm=mock_llm)
        answer = svc.generate_answer("Random question?", [])
        assert isinstance(answer, str)
        assert len(answer) > 0
