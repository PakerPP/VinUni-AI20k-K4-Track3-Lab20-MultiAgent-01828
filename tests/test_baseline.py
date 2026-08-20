"""Tests for the single-agent baseline control arm."""

from multi_agent_research_lab.evaluation.baseline import SingleAgentBaseline
from multi_agent_research_lab.services.llm_client import MockLLMClient
from multi_agent_research_lab.services.search_client import MockSearchClient


def _baseline(retrieval: bool = True) -> SingleAgentBaseline:
    return SingleAgentBaseline(
        llm_client=MockLLMClient(),
        search_client=MockSearchClient() if retrieval else None,
        retrieval=retrieval,
    )


def test_baseline_with_retrieval_receives_evidence() -> None:
    """The fair control arm must get the same evidence the crew gets."""

    state = _baseline(retrieval=True).run("Compare single and multi agent", max_sources=3)
    assert len(state.sources) == 3
    assert state.final_answer
    assert state.route_history == ["single_agent"]


def test_baseline_without_retrieval_has_no_sources() -> None:
    state = _baseline(retrieval=False).run("Compare single and multi agent")
    assert state.sources == []
    assert state.final_answer


def test_baseline_makes_exactly_one_llm_call() -> None:
    """One pass is what makes it a single-agent baseline."""

    llm = MockLLMClient()
    baseline = SingleAgentBaseline(llm_client=llm, search_client=MockSearchClient(), retrieval=True)
    baseline.run("Compare single and multi agent")
    assert llm.call_count == 1


def test_search_failure_is_recorded_and_run_continues() -> None:
    class BrokenSearch(MockSearchClient):
        def search(self, query: str, max_results: int = 5):
            raise RuntimeError("search down")

    baseline = SingleAgentBaseline(
        llm_client=MockLLMClient(), search_client=BrokenSearch(), retrieval=True
    )
    state = baseline.run("Compare single and multi agent")
    assert any("search down" in e for e in state.errors)
    assert state.final_answer  # falls back to the no-evidence prompt
