"""Multi-agent workflow orchestration.

Uses LangGraph when the `[llm]` extra is installed and falls back to an equivalent
plain-Python supervisor loop otherwise, so the lab runs in either environment.
Agent internals stay in `agents/`; only orchestration lives here.
"""

from __future__ import annotations

import logging
from typing import Any

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import (
    ROUTE_ANALYST,
    ROUTE_CRITIC,
    ROUTE_DONE,
    ROUTE_RESEARCHER,
    ROUTE_WRITER,
    SupervisorAgent,
)
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient, MockLLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)

WORKER_ROUTES = (ROUTE_RESEARCHER, ROUTE_ANALYST, ROUTE_WRITER, ROUTE_CRITIC)


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph."""

    def __init__(
        self,
        llm_client: object | None = None,
        search_client: SearchClient | None = None,
        max_iterations: int | None = None,
        enable_critic: bool = True,
        use_langgraph: bool = True,
    ) -> None:
        settings = get_settings()
        self.max_iterations = max_iterations or settings.max_iterations
        self.use_langgraph = use_langgraph

        if llm_client is None:
            llm_client = LLMClient() if settings.openai_api_key else MockLLMClient()
        self.llm_client = llm_client
        self.search_client = search_client or SearchClient()

        self.supervisor = SupervisorAgent(
            max_iterations=self.max_iterations, enable_critic=enable_critic
        )
        self.workers = {
            ROUTE_RESEARCHER: ResearcherAgent(self.search_client, self.llm_client),
            ROUTE_ANALYST: AnalystAgent(self.llm_client, self.search_client),
            ROUTE_WRITER: WriterAgent(self.llm_client),
            ROUTE_CRITIC: CriticAgent(),
        }

    def build(self) -> Any:
        """Create a LangGraph graph with supervisor-driven conditional routing.

        Returns the compiled graph. Typed as `Any` because langgraph is an optional
        dependency - importing its types at module level would make it mandatory.
        """

        from langgraph.graph import END, StateGraph

        builder = StateGraph(ResearchState)

        builder.add_node("supervisor", self.supervisor.run)
        for route in WORKER_ROUTES:
            builder.add_node(route, self.workers[route].run)

        builder.set_entry_point("supervisor")
        builder.add_conditional_edges(
            "supervisor",
            lambda state: state.route_history[-1] if state.route_history else ROUTE_DONE,
            {**{route: route for route in WORKER_ROUTES}, ROUTE_DONE: END},
        )
        # Every worker hands control back to the supervisor.
        for route in WORKER_ROUTES:
            builder.add_edge(route, "supervisor")

        return builder.compile()

    def _run_plain_loop(self, state: ResearchState) -> ResearchState:
        """Supervisor loop without LangGraph — same routing semantics."""

        # +1 so the terminating 'done' decision itself is not counted as work.
        for _ in range(self.max_iterations + 1):
            state = self.supervisor.run(state)
            route = state.route_history[-1]
            if route == ROUTE_DONE:
                break
            state = self.workers[route].run(state)
        return state

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return the final state."""

        with trace_span("workflow.run", {"query": state.request.query}) as span:
            if self.use_langgraph:
                try:
                    graph = self.build()
                    # recursion_limit guards the graph the way max_iterations guards the loop.
                    result = graph.invoke(
                        state, {"recursion_limit": (self.max_iterations + 1) * 2 + 2}
                    )
                    state = (
                        result
                        if isinstance(result, ResearchState)
                        else ResearchState.model_validate(result)
                    )
                    span["attributes"]["engine"] = "langgraph"
                except (ImportError, AttributeError, TypeError) as exc:
                    # Covers langgraph missing *and* a broken/mismatched install.
                    logger.warning("langgraph unavailable (%s), using plain supervisor loop", exc)
                    span["attributes"]["engine"] = "plain-loop"
                    span["attributes"]["langgraph_error"] = str(exc)
                    state = self._run_plain_loop(state)
            else:
                span["attributes"]["engine"] = "plain-loop"
                state = self._run_plain_loop(state)

            span["attributes"]["iterations"] = state.iteration
            state.add_trace_event(
                "workflow.done",
                {
                    "iterations": state.iteration,
                    "route_history": state.route_history,
                    "errors": state.errors,
                },
            )
        return state
