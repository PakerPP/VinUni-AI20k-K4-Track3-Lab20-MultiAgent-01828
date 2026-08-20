"""Writer agent implementation."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.researcher import _usage_of
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Writer agent in a multi-agent research system.
Turn the analysis into a clear, well-structured answer for the stated audience.
Rules:
- Cite claims inline using the [source_id] labels you were given.
- Never introduce a source id that does not appear in the evidence list.
- Mark any claim resting on synthetic evidence as (synthetic benchmark evidence).
- Structure: short intro, 3-5 substantive sections with headings, brief conclusion.
- No hedging filler, no restating the prompt."""


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: object | None = None) -> None:
        self.llm_client = llm_client

    def _build_prompt(self, state: ResearchState) -> str:
        allowed = ", ".join(f"[{d.metadata.get('source_id')}]" for d in state.sources)
        return (
            f"Question: {state.request.query}\n"
            f"Audience: {state.request.audience}\n\n"
            f"Research notes:\n{state.research_notes}\n\n"
            f"Analysis:\n{state.analysis_notes}\n\n"
            f"Allowed citation ids: {allowed}\n\n"
            "Write the final answer now."
        )

    def _fallback_answer(self, state: ResearchState) -> str:
        cites = " ".join(f"[{d.metadata.get('source_id')}]" for d in state.sources)
        return (
            f"# {state.request.query}\n\n"
            "## Summary\n"
            f"{(state.analysis_notes or '')[:1200]}\n\n"
            "## Evidence base\n"
            + "\n".join(f"- {d.title} [{d.metadata.get('source_id')}]" for d in state.sources)
            + f"\n\nSources cited: {cites}\n"
        )

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""

        if not state.analysis_notes:
            state.errors.append(f"{self.name}: missing analysis notes")
            state.add_trace_event("writer.error", {"error": "missing analysis notes"})
            return state

        with trace_span("writer.run") as span:
            answer = None
            usage: dict[str, object] = {}
            if self.llm_client is not None:
                try:
                    response = self.llm_client.complete(  # type: ignore[attr-defined]
                        SYSTEM_PROMPT, self._build_prompt(state)
                    )
                    answer = response.content
                    usage = _usage_of(response)
                except AgentExecutionError as exc:
                    state.errors.append(f"{self.name}: llm failed: {exc}")
                    logger.warning("writer.llm_failed falling back error=%s", exc)

            if not answer:
                answer = self._fallback_answer(state)

            state.final_answer = answer
            span["attributes"]["answer_chars"] = len(answer)

            state.agent_results.append(
                AgentResult(
                    agent=AgentName.WRITER,
                    content=answer,
                    metadata={"answer_chars": len(answer), **usage},
                )
            )
            state.add_trace_event("writer.done", {"answer_chars": len(answer)})
        return state
