"""Critic agent: citation and grounding verification."""

from __future__ import annotations

import logging
import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[([A-Za-z0-9_\-]{1,40})\]")


class CriticAgent(BaseAgent):
    """Fact-checks citation usage in the final answer.

    This runs deterministically rather than through an LLM: hallucinated citations
    are exactly the failure mode the corpus benchmark targets, and a string check
    detects them without adding cost or a second hallucination surface.
    """

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings."""

        if not state.final_answer:
            state.errors.append(f"{self.name}: no final answer to review")
            return state

        with trace_span("critic.run") as span:
            valid_ids = {
                str(d.metadata.get("source_id"))
                for d in state.sources
                if d.metadata.get("source_id")
            }
            synthetic_ids = {
                str(d.metadata.get("source_id"))
                for d in state.sources
                if d.metadata.get("is_synthetic")
            }

            cited = set(_CITATION_RE.findall(state.final_answer))
            # Ignore bracketed words that were never meant as citations.
            cited = {c for c in cited if c in valid_ids or c.lower() not in {"synthetic", "sic"}}

            hallucinated = sorted(cited - valid_ids)
            used = sorted(cited & valid_ids)
            uncited = sorted(valid_ids - cited)
            synth_used = sorted(set(used) & synthetic_ids)

            coverage = len(used) / len(valid_ids) if valid_ids else 0.0

            findings = [
                f"Citation coverage: {coverage:.0%} ({len(used)}/{len(valid_ids)} sources cited).",
            ]
            if hallucinated:
                findings.append(f"FAIL - citations not in evidence set: {', '.join(hallucinated)}.")
                state.errors.append(
                    f"{self.name}: hallucinated citations: {', '.join(hallucinated)}"
                )
            else:
                findings.append("PASS - every citation maps to a supplied source.")
            if uncited:
                findings.append(f"Unused sources: {', '.join(uncited)}.")
            if synth_used:
                findings.append(
                    f"Synthetic evidence relied upon: {', '.join(synth_used)} "
                    "- must be labelled as benchmark-fictional."
                )

            report = "\n".join(f"- {line}" for line in findings)
            span["attributes"]["citation_coverage"] = coverage

            state.agent_results.append(
                AgentResult(
                    agent=AgentName.CRITIC,
                    content=report,
                    metadata={
                        "citation_coverage": coverage,
                        "hallucinated": hallucinated,
                        "cited": used,
                        "synthetic_used": synth_used,
                    },
                )
            )
            state.add_trace_event(
                "critic.done",
                {"citation_coverage": coverage, "hallucinated": len(hallucinated)},
            )
            logger.info("critic.done coverage=%.2f hallucinated=%d", coverage, len(hallucinated))
        return state
