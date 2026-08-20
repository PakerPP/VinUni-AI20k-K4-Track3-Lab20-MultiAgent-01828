"""Supervisor / router implementation."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

ROUTE_RESEARCHER = "researcher"
ROUTE_ANALYST = "analyst"
ROUTE_WRITER = "writer"
ROUTE_CRITIC = "critic"
ROUTE_DONE = "done"

# A worker that failed this many times is skipped rather than retried forever.
MAX_FAILURES_PER_AGENT = 2


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop.

    Routing is deliberately state-driven rather than LLM-driven: the decision is
    cheap, deterministic, and easy to explain in a trace — which is what the lab
    rubric asks for. The policy fills missing state fields in dependency order.
    """

    name = "supervisor"

    def __init__(
        self,
        max_iterations: int | None = None,
        enable_critic: bool = True,
    ) -> None:
        settings = get_settings()
        self.max_iterations = max_iterations or settings.max_iterations
        self.enable_critic = enable_critic

    def _failure_count(self, state: ResearchState, agent: str) -> int:
        return sum(1 for err in state.errors if err.startswith(f"{agent}:"))

    def _blocked(self, state: ResearchState, agent: str) -> bool:
        return self._failure_count(state, agent) >= MAX_FAILURES_PER_AGENT

    def decide(self, state: ResearchState) -> str:
        """Return the next route without mutating state."""

        # Guardrail: never let the graph spin forever.
        if state.iteration >= self.max_iterations:
            logger.warning("supervisor.max_iterations_reached iteration=%s", state.iteration)
            return ROUTE_DONE

        # 1. No evidence yet -> gather it (unless the researcher keeps failing).
        if not state.sources and not self._blocked(state, ROUTE_RESEARCHER):
            return ROUTE_RESEARCHER

        # 2. Evidence but no analysis -> analyze.
        if not state.analysis_notes and not self._blocked(state, ROUTE_ANALYST):
            return ROUTE_ANALYST

        # 3. Analysis but no answer -> write.
        if not state.final_answer and not self._blocked(state, ROUTE_WRITER):
            return ROUTE_WRITER

        # 4. Answer exists -> one optional verification pass.
        if (
            self.enable_critic
            and state.final_answer
            and ROUTE_CRITIC not in state.route_history
            and not self._blocked(state, ROUTE_CRITIC)
        ):
            return ROUTE_CRITIC

        return ROUTE_DONE

    def run(self, state: ResearchState) -> ResearchState:
        """Record the next route on shared state."""

        route = self.decide(state)
        state.record_route(route)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.SUPERVISOR,
                content=f"route={route}",
                metadata={"iteration": state.iteration, "route": route},
            )
        )
        state.add_trace_event(
            "supervisor.route",
            {
                "route": route,
                "iteration": state.iteration,
                "has_sources": bool(state.sources),
                "has_analysis": bool(state.analysis_notes),
                "has_answer": bool(state.final_answer),
                "errors": len(state.errors),
            },
        )
        logger.info("supervisor.route route=%s iteration=%s", route, state.iteration)
        return state
