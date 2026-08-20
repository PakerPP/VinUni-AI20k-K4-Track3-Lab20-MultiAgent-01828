"""LLM-as-judge quality scoring.

Replaces the earlier heuristic score, which rewarded length and headings — a verbose
answer with tidy structure outscored a sharper short one.

Caveat that belongs in any report using this: the judge runs on the same model family
as the agents, so it is a *relative* signal for comparing two arms of one benchmark,
not an absolute measure of quality. Peer review stays the ground truth.
"""

from __future__ import annotations

import logging
import re

from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import QualityVerdict
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

JUDGE_SYSTEM = """You grade research answers for a technical audience. Judge on:
1. Substance - does it actually answer the question with specifics?
2. Grounding - are claims tied to the cited evidence rather than asserted?
3. Structure - is it organised and readable?
4. Honesty - does it acknowledge uncertainty and limits instead of overclaiming?

Do not reward length. A short precise answer beats a padded one.

Reply in exactly this format, nothing else:
SCORE: <number 0-10, one decimal>
STRENGTHS: <one line>
WEAKNESSES: <one line>"""

# Accepts a leading minus so an out-of-range reply clamps instead of parsing as None.
_SCORE_RE = re.compile(r"SCORE:\s*(-?[0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_STRENGTHS_RE = re.compile(r"STRENGTHS:\s*(.+)", re.IGNORECASE)
_WEAKNESSES_RE = re.compile(r"WEAKNESSES:\s*(.+)", re.IGNORECASE)


def parse_verdict(raw: str) -> QualityVerdict | None:
    """Parse the judge's three-line reply, tolerating extra prose around it."""

    match = _SCORE_RE.search(raw)
    if not match:
        logger.warning("judge.unparseable verdict=%r", raw[:120])
        return None
    score = max(0.0, min(10.0, float(match.group(1))))
    strengths = _STRENGTHS_RE.search(raw)
    weaknesses = _WEAKNESSES_RE.search(raw)
    return QualityVerdict(
        score=score,
        strengths=strengths.group(1).strip() if strengths else "",
        weaknesses=weaknesses.group(1).strip() if weaknesses else "",
    )


class AnswerJudge:
    """Grades an answer 0-10 against the lab rubric."""

    def __init__(self, llm_client: object) -> None:
        self.llm_client = llm_client

    def score(self, state: ResearchState) -> QualityVerdict | None:
        """Grade `state.final_answer`; returns None if the judge is unavailable."""

        answer = (state.final_answer or "").strip()
        if not answer:
            return QualityVerdict(score=0.0, weaknesses="empty answer")

        cited = ", ".join(
            f"[{d.metadata.get('source_id')}]" for d in state.sources if d.metadata.get("source_id")
        )
        prompt = (
            f"Question: {state.request.query}\n"
            f"Audience: {state.request.audience}\n"
            f"Evidence ids available: {cited or '(none - no retrieval)'}\n\n"
            f"Answer to grade:\n{answer}"
        )
        try:
            raw = self.llm_client.complete(JUDGE_SYSTEM, prompt).content  # type: ignore[attr-defined]
        except AgentExecutionError as exc:
            logger.warning("judge.call_failed error=%s", exc)
            return None
        return parse_verdict(raw)
