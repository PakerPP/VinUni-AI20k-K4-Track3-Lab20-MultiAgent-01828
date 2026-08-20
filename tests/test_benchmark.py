"""Tests for benchmark metrics."""

from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import (
    compute_citation_coverage,
    count_hallucinated_citations,
    run_benchmark,
    score_quality,
)


def _state(answer: str | None = None, ids: list[str] | None = None) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.sources = [
        SourceDocument(title=f"D{i}", snippet="s", metadata={"source_id": i}) for i in (ids or [])
    ]
    state.final_answer = answer
    return state


def test_citation_coverage_counts_only_valid_ids() -> None:
    assert compute_citation_coverage(_state("uses [a] and [zzz]", ["a", "b"])) == 0.5


def test_citation_coverage_zero_without_answer() -> None:
    assert compute_citation_coverage(_state(None, ["a"])) == 0.0


def test_counts_hallucinated_citations() -> None:
    assert count_hallucinated_citations(_state("[a] [ghost] [phantom]", ["a"])) == 2


def test_quality_penalises_hallucinations() -> None:
    grounded = _state("# H\n\n- point [a]\n" + "word " * 200, ["a"])
    fabricated = _state("# H\n\n- point [ghost]\n" + "word " * 200, ["a"])
    assert score_quality(grounded) > score_quality(fabricated)


def test_quality_zero_for_empty_answer() -> None:
    assert score_quality(_state(None, ["a"])) == 0.0


def test_run_benchmark_captures_failure_without_crashing() -> None:
    def _boom(_query: str) -> ResearchState:
        raise RuntimeError("provider down")

    state, metrics = run_benchmark("failing", "some query", _boom)
    assert metrics.failure_rate == 1.0
    assert any("provider down" in e for e in state.errors)


def test_run_benchmark_measures_successful_run() -> None:
    def _ok(query: str) -> ResearchState:
        return _state("# Answer\n\n- grounded [a]", ["a"])

    _, metrics = run_benchmark("ok", "q", _ok)
    assert metrics.failure_rate == 0.0
    assert metrics.citation_coverage == 1.0
    assert metrics.latency_seconds >= 0
