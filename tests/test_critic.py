"""Tests for citation verification."""

from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState


def _state(answer: str, ids: list[str], synthetic: list[str] | None = None) -> ResearchState:
    synthetic = synthetic or []
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.sources = [
        SourceDocument(
            title=f"Doc {i}",
            snippet="snippet",
            metadata={"source_id": i, "is_synthetic": i in synthetic},
        )
        for i in ids
    ]
    state.final_answer = answer
    return state


def test_detects_hallucinated_citation() -> None:
    state = CriticAgent().run(_state("Claim [ghost].", ["real1"]))
    result = state.agent_results[-1]
    assert result.metadata["hallucinated"] == ["ghost"]
    assert any("hallucinated" in e for e in state.errors)


def test_passes_when_all_citations_valid() -> None:
    state = CriticAgent().run(_state("Claim [a1] and [a2].", ["a1", "a2"]))
    result = state.agent_results[-1]
    assert result.metadata["hallucinated"] == []
    assert result.metadata["citation_coverage"] == 1.0


def test_partial_coverage_is_reported() -> None:
    state = CriticAgent().run(_state("Only [a1] cited.", ["a1", "a2"]))
    assert state.agent_results[-1].metadata["citation_coverage"] == 0.5


def test_flags_reliance_on_synthetic_evidence() -> None:
    state = CriticAgent().run(_state("Per [syn1].", ["syn1"], synthetic=["syn1"]))
    assert state.agent_results[-1].metadata["synthetic_used"] == ["syn1"]


def test_no_answer_records_error() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state = CriticAgent().run(state)
    assert any("no final answer" in e for e in state.errors)
