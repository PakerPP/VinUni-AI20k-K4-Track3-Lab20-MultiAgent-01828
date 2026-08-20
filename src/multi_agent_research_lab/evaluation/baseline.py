"""Single-agent baseline: the control arm of the benchmark.

By default the baseline receives the **same retrieved evidence** as the multi-agent
crew and does research, analysis, and writing in one LLM call with one prompt. That
isolates the variable the lab actually asks about: orchestration.

An earlier version skipped retrieval entirely. It made multi-agent look dramatically
better (citation coverage 0% vs 87%), but the gap was measuring *retrieval*, not
orchestration — a baseline with nothing to cite cannot cite anything. `retrieval=False`
keeps that naive arm available for contrast, and the benchmark reports both.
"""

from __future__ import annotations

import logging

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient, MockLLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)

# Mirrors the combined duties of Researcher + Analyst + Writer in a single prompt.
SYSTEM_PROMPT = """You are a research assistant working alone. Using ONLY the evidence
supplied, research the question, weigh the evidence, and write the final answer in one pass.
Rules:
- Cite claims inline using the [source_id] labels you were given.
- Never introduce a source id that does not appear in the evidence list.
- Square brackets are RESERVED for citation ids only.
- Mark any claim resting on synthetic evidence as (synthetic benchmark evidence).
- Structure: short intro, 3-5 sections with headings, brief conclusion."""

NO_RETRIEVAL_PROMPT = """You are a research assistant. Answer the question thoroughly and
in a well-structured way for a technical audience. Use headings and bullet points."""


class SingleAgentBaseline:
    """One-shot baseline runner with the same state contract as the workflow."""

    def __init__(
        self,
        llm_client: object | None = None,
        search_client: SearchClient | None = None,
        retrieval: bool = True,
    ) -> None:
        settings = get_settings()
        if llm_client is None:
            llm_client = LLMClient() if settings.openai_api_key else MockLLMClient()
        self.llm_client = llm_client
        self.retrieval = retrieval
        self.search_client = search_client or (SearchClient() if retrieval else None)

    def _build_prompt(self, state: ResearchState) -> str:
        blocks = []
        for doc in state.sources:
            meta = doc.metadata
            flag = " (synthetic)" if meta.get("is_synthetic") else ""
            body = str(meta.get("full_text") or doc.snippet)[:2500]
            blocks.append(f"[{meta.get('source_id')}] {doc.title}{flag}\n{body}")
        evidence = "\n\n---\n\n".join(blocks)
        allowed = ", ".join(f"[{d.metadata.get('source_id')}]" for d in state.sources)
        return (
            f"Research question: {state.request.query}\n"
            f"Audience: {state.request.audience}\n\n"
            f"Evidence:\n\n{evidence}\n\n"
            f"Allowed citation ids: {allowed}\n\n"
            "Write the final answer now."
        )

    def run(self, query: str, max_sources: int = 5) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=query, max_sources=max_sources))
        with trace_span("baseline.run", {"query": query, "retrieval": self.retrieval}) as span:
            state.record_route("single_agent")

            # Same retrieval step the crew gets — one search, no iteration.
            if self.retrieval and self.search_client is not None:
                try:
                    state.sources = self.search_client.search(
                        query, max_results=state.request.max_sources
                    )
                    state.add_trace_event("baseline.search", {"num_sources": len(state.sources)})
                except Exception as exc:
                    state.errors.append(f"baseline: search failed: {exc}")
                    logger.warning("baseline.search_failed error=%s", exc)

            if state.sources:
                system, user = SYSTEM_PROMPT, self._build_prompt(state)
            else:
                system, user = NO_RETRIEVAL_PROMPT, state.request.query

            try:
                response = self.llm_client.complete(system, user)  # type: ignore[attr-defined]
                state.final_answer = response.content
                state.agent_results.append(
                    AgentResult(
                        agent=AgentName.WRITER,
                        content=response.content,
                        metadata={
                            "mode": "single_agent",
                            "retrieval": self.retrieval,
                            "num_sources": len(state.sources),
                            "input_tokens": response.input_tokens,
                            "output_tokens": response.output_tokens,
                            "cost_usd": response.cost_usd,
                        },
                    )
                )
                span["attributes"]["answer_chars"] = len(response.content)
            except AgentExecutionError as exc:
                state.errors.append(f"baseline: {exc}")
                logger.error("baseline.failed error=%s", exc)

            state.add_trace_event(
                "baseline.done",
                {"has_answer": bool(state.final_answer), "num_sources": len(state.sources)},
            )
        return state
