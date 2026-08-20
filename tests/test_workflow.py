"""End-to-end workflow tests using mock services (no API key, no corpus)."""

from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import MockLLMClient
from multi_agent_research_lab.services.search_client import MockSearchClient


def _workflow(**kwargs) -> MultiAgentWorkflow:
    return MultiAgentWorkflow(
        llm_client=MockLLMClient(),
        search_client=MockSearchClient(),
        **kwargs,
    )


def _state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Compare single and multi agent systems"))


def test_plain_loop_runs_all_stages_in_order() -> None:
    state = _workflow(use_langgraph=False).run(_state())
    assert state.route_history == ["researcher", "analyst", "writer", "critic", "done"]
    assert state.sources and state.research_notes
    assert state.analysis_notes and state.final_answer
    assert state.errors == []


def test_langgraph_engine_produces_same_final_state() -> None:
    """LangGraph and the fallback loop must agree; skips if langgraph is unusable."""

    plain = _workflow(use_langgraph=False).run(_state())
    graph = _workflow(use_langgraph=True).run(_state())
    assert graph.route_history == plain.route_history
    assert bool(graph.final_answer) == bool(plain.final_answer)


def test_workflow_respects_max_iterations() -> None:
    state = _workflow(use_langgraph=False, max_iterations=2).run(_state())
    assert state.iteration <= 3  # 2 work steps + terminating 'done'
    assert state.route_history[-1] == "done"


def test_trace_events_recorded_for_each_stage() -> None:
    state = _workflow(use_langgraph=False).run(_state())
    names = {e["name"] for e in state.trace}
    assert {"researcher.done", "analyst.done", "writer.done", "critic.done"} <= names


def test_search_failure_is_recorded_and_does_not_crash() -> None:
    class BrokenSearch(MockSearchClient):
        def search(self, query: str, max_results: int = 5):
            raise RuntimeError("search backend down")

    workflow = MultiAgentWorkflow(
        llm_client=MockLLMClient(), search_client=BrokenSearch(), use_langgraph=False
    )
    state = workflow.run(_state())
    assert any("search backend down" in e for e in state.errors)
    assert state.route_history[-1] == "done"
