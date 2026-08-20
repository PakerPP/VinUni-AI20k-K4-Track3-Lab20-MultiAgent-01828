"""Researcher agent implementation."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Research agent in a multi-agent research system.
Your only job is to condense retrieved evidence into factual research notes.
Rules:
- Use ONLY the supplied evidence. Never invent sources, numbers, or citations.
- Attribute every note with its [source_id].
- Mark synthetic evidence with the word (synthetic) in parentheses, never in brackets.
- Square brackets are RESERVED for citation ids only.
- Be concise: bullet points, no preamble, no conclusions."""


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self, search_client: SearchClient, llm_client: object | None = None) -> None:
        self.search_client = search_client
        self.llm_client = llm_client

    def _build_prompt(self, state: ResearchState) -> str:
        blocks = []
        for doc in state.sources:
            meta = doc.metadata
            flag = " (synthetic)" if meta.get("is_synthetic") else ""
            body = str(meta.get("full_text") or doc.snippet)[:2500]
            blocks.append(f"[{meta.get('source_id')}] {doc.title}{flag}\n{body}")
        evidence = "\n\n---\n\n".join(blocks)
        return (
            f"Research question: {state.request.query}\n"
            f"Audience: {state.request.audience}\n\n"
            f"Evidence:\n\n{evidence}\n\n"
            "Write 8-12 research note bullets, each ending with its [source_id]."
        )

    def _fallback_notes(self, state: ResearchState) -> str:
        """Deterministic notes if no LLM is available — keeps the pipeline running."""

        lines = []
        for doc in state.sources:
            meta = doc.metadata
            flag = " (synthetic)" if meta.get("is_synthetic") else ""
            lines.append(f"- {doc.title}{flag}: {doc.snippet[:220]} [{meta.get('source_id')}]")
        return "\n".join(lines)

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""

        with trace_span("researcher.run", {"query": state.request.query}) as span:
            try:
                docs = self.search_client.search(
                    state.request.query, max_results=state.request.max_sources
                )
            except Exception as exc:
                state.errors.append(f"{self.name}: search failed: {exc}")
                state.add_trace_event("researcher.error", {"error": str(exc)})
                logger.error("researcher.search_failed error=%s", exc)
                return state

            if not docs:
                state.errors.append(f"{self.name}: no sources found")
                state.add_trace_event("researcher.error", {"error": "no sources"})
                return state

            state.sources = docs

            notes = None
            if self.llm_client is not None:
                try:
                    response = self.llm_client.complete(  # type: ignore[attr-defined]
                        SYSTEM_PROMPT, self._build_prompt(state)
                    )
                    notes = response.content
                except AgentExecutionError as exc:
                    state.errors.append(f"{self.name}: llm failed: {exc}")
                    logger.warning("researcher.llm_failed falling back error=%s", exc)

            if not notes:
                notes = self._fallback_notes(state)

            state.research_notes = notes
            span["attributes"]["num_sources"] = len(docs)

            state.agent_results.append(
                AgentResult(
                    agent=AgentName.RESEARCHER,
                    content=notes,
                    metadata={
                        "num_sources": len(docs),
                        "source_ids": [d.metadata.get("source_id") for d in docs],
                        "topic_id": docs[0].metadata.get("topic_id"),
                    },
                )
            )
            state.add_trace_event(
                "researcher.done",
                {"num_sources": len(docs), "notes_chars": len(notes)},
            )
        return state
