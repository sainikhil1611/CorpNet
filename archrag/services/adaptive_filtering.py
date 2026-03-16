"""Service: Adaptive Filtering-based Generation.

Corresponds to ArchRAG §3.2 — Equations (1) and (2).
For each layer's results, the LLM extracts an analysis report
with relevance scores, then merges all reports into a final answer.
"""

from __future__ import annotations

import json
import logging

from archrag.domain.models import AnalysisPoint, AnalysisReport, SearchResult
from archrag.ports.llm import LLMPort
from archrag.prompts.filtering import (
    FILTER_PROMPT,
    FILTER_SYSTEM,
    MERGE_PROMPT,
    MERGE_SYSTEM,
)

log = logging.getLogger(__name__)


class AdaptiveFilteringService:
    """Filter and merge search results into a final answer."""

    def __init__(self, llm: LLMPort, *, response_format: str = "A concise, direct answer (1-3 paragraphs)."):
        self._llm = llm
        self._response_format = response_format

    # ── public ──

    def generate_answer(
        self,
        question: str,
        results_per_layer: list[list[SearchResult]],
    ) -> str:
        """Run the two-stage adaptive filtering pipeline.

        Stage 1 (Eq. 1): For each layer i, ``A_i = LLM(P_filter || R_i)``
        Stage 2 (Eq. 2): ``Output = LLM(P_merge || Sort({A_0, ..., A_n}))``
        """
        # Stage 1: Extract analysis reports per layer
        reports: list[AnalysisReport] = []
        for layer_results in results_per_layer:
            if not layer_results:
                continue
            report = self._extract_report(question, layer_results)
            reports.append(report)

        if not reports:
            return self._llm.generate(
                f"Answer the following question:\n{question}",
                system="You are a helpful assistant.",
            )

        # Stage 2: Sort all points by score, merge into final answer
        all_points: list[AnalysisPoint] = []
        for report in reports:
            all_points.extend(report.points)
        all_points.sort(key=lambda p: p.score, reverse=True)

        report_text = "\n".join(
            f"- [Score {p.score}] {p.description}" for p in all_points
        )

        merge_prompt = MERGE_PROMPT.format(
            response_format=self._response_format,
            question=question,
            report_data=report_text,
        )

        answer = self._llm.generate(merge_prompt, system=MERGE_SYSTEM)
        return answer

    # ── private helpers ──

    def _extract_report(
        self,
        question: str,
        layer_results: list[SearchResult],
    ) -> AnalysisReport:
        """Run the filter prompt on one layer's results → AnalysisReport."""
        context_data = "\n\n".join(
            f"### Item {i+1} (layer {r.level}, distance {r.distance:.4f})\n{r.text}"
            for i, r in enumerate(layer_results)
        )

        prompt = FILTER_PROMPT.format(
            question=question,
            context_data=context_data,
        )

        try:
            result = self._llm.generate_json(prompt, system=FILTER_SYSTEM)
            points = [
                AnalysisPoint(
                    description=p.get("description", ""),
                    score=float(p.get("score", 0)),
                )
                for p in result.get("points", [])
            ]
        except (json.JSONDecodeError, Exception) as exc:
            log.warning("Filter extraction failed: %s", exc)
            # Fallback: treat all text as a single point
            all_text = " ".join(r.text for r in layer_results)
            points = [AnalysisPoint(description=all_text[:2000], score=50.0)]

        level = layer_results[0].level if layer_results else 0
        return AnalysisReport(level=level, points=points)
