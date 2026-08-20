"""Analyst agent implementation."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an Analyst agent in a multi-agent research system.
You receive research notes and must turn them into structured insight.
Produce exactly these sections:
1. Key claims (with [source_id])
2. Points of agreement
3. Tensions or contradictions
4. Weak evidence (call out synthetic sources explicitly)
5. What is still missing
Rules: reason only over the supplied notes; never introduce new sources.
Square brackets are RESERVED for citation ids from the evidence list."""


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(
        self,
        llm_client: object | None = None,
        search_client: object | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.search_client = search_client

    def _build_prompt(self, state: ResearchState) -> str:
        facts_block = ""
        if self.search_client is not None:
            try:
                facts = self.search_client.load_facts(state.request.query)  # type: ignore[attr-defined]
                if facts:
                    rendered = "\n".join(
                        f"- (fact {f.get('fact_id')}) {f.get('statement')} "
                        f"(confidence={f.get('confidence')})"
                        for f in facts
                    )
                    facts_block = f"\n\nCorpus fact bank:\n{rendered}"
            except Exception as exc:  # fact bank is an enrichment, never fatal
                logger.warning("analyst.fact_bank_unavailable error=%s", exc)

        synthetic = [
            str(d.metadata.get("source_id"))
            for d in state.sources
            if d.metadata.get("is_synthetic")
        ]
        synth_note = (
            f"\n\nSynthetic (fictional) source ids: {', '.join(synthetic)}"
            if synthetic
            else "\n\nNo synthetic sources present."
        )
        return (
            f"Research question: {state.request.query}\n\n"
            f"Research notes:\n{state.research_notes}"
            f"{facts_block}{synth_note}\n\nWrite the five-section analysis."
        )

    def _fallback_analysis(self, state: ResearchState) -> str:
        real = [d for d in state.sources if not d.metadata.get("is_synthetic")]
        synth = [d for d in state.sources if d.metadata.get("is_synthetic")]
        lines = [
            "1. Key claims",
            *(f"   - {d.title}: {d.snippet[:160]} [{d.metadata.get('source_id')}]" for d in real),
            "2. Points of agreement",
            f"   - {len(real)} non-synthetic sources address the question directly.",
            "3. Tensions or contradictions",
            "   - Not assessed without an LLM; review claims manually.",
            "4. Weak evidence",
            *(
                [f"   - [synthetic] {d.metadata.get('source_id')}: {d.title}" for d in synth]
                or ["   - No synthetic sources flagged."]
            ),
            "5. What is still missing",
            "   - Quantitative comparison across sources.",
        ]
        return "\n".join(lines)

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""

        if not state.research_notes:
            state.errors.append(f"{self.name}: missing research notes")
            state.add_trace_event("analyst.error", {"error": "missing research notes"})
            return state

        with trace_span("analyst.run") as span:
            analysis = None
            if self.llm_client is not None:
                try:
                    response = self.llm_client.complete(  # type: ignore[attr-defined]
                        SYSTEM_PROMPT, self._build_prompt(state)
                    )
                    analysis = response.content
                except AgentExecutionError as exc:
                    state.errors.append(f"{self.name}: llm failed: {exc}")
                    logger.warning("analyst.llm_failed falling back error=%s", exc)

            if not analysis:
                analysis = self._fallback_analysis(state)

            state.analysis_notes = analysis
            span["attributes"]["analysis_chars"] = len(analysis)

            state.agent_results.append(
                AgentResult(
                    agent=AgentName.ANALYST,
                    content=analysis,
                    metadata={"num_sources_considered": len(state.sources)},
                )
            )
            state.add_trace_event("analyst.done", {"analysis_chars": len(analysis)})
        return state
