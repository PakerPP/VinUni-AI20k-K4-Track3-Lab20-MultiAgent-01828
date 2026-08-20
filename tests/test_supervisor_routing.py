"""Unit tests for the supervisor routing policy.

Replaces the original skeleton guard test (`test_agents_todo.py`), which asserted the
TODO was still unimplemented and is obsolete now that routing exists.
"""

from multi_agent_research_lab.agents.supervisor import (
    ROUTE_ANALYST,
    ROUTE_CRITIC,
    ROUTE_DONE,
    ROUTE_RESEARCHER,
    ROUTE_WRITER,
    SupervisorAgent,
)
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState


def _state(**kwargs) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    for key, value in kwargs.items():
        setattr(state, key, value)
    return state


def _source(source_id: str = "s1") -> SourceDocument:
    return SourceDocument(title="T", snippet="S", metadata={"source_id": source_id})


def test_routes_to_researcher_when_no_sources() -> None:
    assert SupervisorAgent().decide(_state()) == ROUTE_RESEARCHER


def test_routes_to_analyst_when_sources_but_no_analysis() -> None:
    assert SupervisorAgent().decide(_state(sources=[_source()])) == ROUTE_ANALYST


def test_routes_to_writer_when_analysis_ready() -> None:
    state = _state(sources=[_source()], analysis_notes="notes")
    assert SupervisorAgent().decide(state) == ROUTE_WRITER


def test_routes_to_critic_after_answer() -> None:
    state = _state(sources=[_source()], analysis_notes="a", final_answer="answer")
    assert SupervisorAgent().decide(state) == ROUTE_CRITIC


def test_done_after_critic_has_run() -> None:
    state = _state(
        sources=[_source()],
        analysis_notes="a",
        final_answer="answer",
        route_history=[ROUTE_CRITIC],
    )
    assert SupervisorAgent().decide(state) == ROUTE_DONE


def test_critic_can_be_disabled() -> None:
    state = _state(sources=[_source()], analysis_notes="a", final_answer="answer")
    assert SupervisorAgent(enable_critic=False).decide(state) == ROUTE_DONE


def test_max_iterations_guard_stops_the_loop() -> None:
    """The iteration guard must win even when required fields are still missing."""

    state = _state(iteration=6)
    assert SupervisorAgent(max_iterations=6).decide(state) == ROUTE_DONE


def test_repeatedly_failing_agent_is_skipped() -> None:
    """Two researcher failures -> stop retrying and move on instead of looping."""

    state = _state(errors=["researcher: boom", "researcher: boom"])
    assert SupervisorAgent().decide(state) == ROUTE_ANALYST


def test_run_records_route_and_trace() -> None:
    state = _state()
    state = SupervisorAgent().run(state)
    assert state.route_history == [ROUTE_RESEARCHER]
    assert state.iteration == 1
    assert state.trace[-1]["name"] == "supervisor.route"
