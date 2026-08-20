"""Benchmark: single-agent baseline vs multi-agent workflow."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.core.schemas import AgentName, BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.judge import AnswerJudge

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]

_CITATION_RE = re.compile(r"\[([A-Za-z0-9_\-]{1,40})\]")


def compute_citation_coverage(state: ResearchState) -> float:
    """Fraction of retrieved sources actually cited in the final answer."""

    if not state.sources or not state.final_answer:
        return 0.0
    valid = {str(d.metadata.get("source_id")) for d in state.sources if d.metadata.get("source_id")}
    if not valid:
        return 0.0
    cited = set(_CITATION_RE.findall(state.final_answer)) & valid
    return len(cited) / len(valid)


def count_hallucinated_citations(state: ResearchState) -> int:
    """Citation ids in the answer that do not exist in the evidence set."""

    if not state.final_answer:
        return 0
    valid = {str(d.metadata.get("source_id")) for d in state.sources if d.metadata.get("source_id")}
    cited = set(_CITATION_RE.findall(state.final_answer))
    return len({c for c in cited if c not in valid and c.lower() not in {"synthetic", "sic"}})


def score_quality(state: ResearchState) -> float:
    """Heuristic 0-10 quality proxy.

    Deliberately mechanical: it rewards grounding and structure, which peer review
    can then override. It is a screening signal, not a replacement for the rubric.
    """

    answer = state.final_answer or ""
    if not answer:
        return 0.0

    score = 0.0
    # Substance (0-3): length bands rather than raw length.
    words = len(answer.split())
    score += 3.0 if words >= 600 else 2.0 if words >= 300 else 1.0 if words >= 120 else 0.0
    # Grounding (0-4): citation coverage.
    score += 4.0 * compute_citation_coverage(state)
    # Structure (0-2): headings and bullets.
    score += 1.0 if re.search(r"^#{1,3} ", answer, re.MULTILINE) else 0.0
    score += 1.0 if re.search(r"^\s*[-*] ", answer, re.MULTILINE) else 0.0
    # Penalty (up to -3): fabricated citations.
    score -= min(3.0, 1.5 * count_hallucinated_citations(state))
    # Penalty: unrecovered agent errors.
    score -= min(2.0, 0.5 * len(state.errors))
    return max(0.0, min(10.0, score))


def estimate_state_cost(state: ResearchState, llm_client: object | None = None) -> float | None:
    """Total USD spent producing this state, if the client tracked it."""

    if llm_client is not None and hasattr(llm_client, "total_cost_usd"):
        return float(llm_client.total_cost_usd)
    return None


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
    llm_client: object | None = None,
    judge: AnswerJudge | None = None,
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Run one query and collect latency, cost, quality, citation and failure metrics.

    `judge` grades the answer with an LLM when supplied; otherwise the heuristic
    `score_quality` stands in. The heuristic is a screening signal only - it rewards
    length and structure, which a judge does not.
    """

    started = perf_counter()
    failed = False
    try:
        state = runner(query)
    except Exception as exc:
        logger.error("benchmark.run_failed run=%s error=%s", run_name, exc)
        failed = True
        state = ResearchState(request=ResearchQuery(query=query))
        state.errors.append(f"run failed: {exc}")
    latency = perf_counter() - started

    # Ensure citation stats exist even when the workflow skipped the critic.
    if state.final_answer and not any(r.agent == AgentName.CRITIC for r in state.agent_results):
        CriticAgent().run(state)

    quality = score_quality(state)
    grader = "heuristic"
    verdict = None
    if judge is not None and state.final_answer:
        verdict = judge.score(state)
        if verdict is not None:
            quality = verdict.score
            grader = "llm-judge"

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=estimate_state_cost(state, llm_client),
        quality_score=quality,
        citation_coverage=compute_citation_coverage(state),
        failure_rate=1.0 if failed else 0.0,
        notes=(
            f"routes={'>'.join(state.route_history) or 'n/a'}; "
            f"sources={len(state.sources)}; "
            f"llm_calls={getattr(llm_client, 'call_count', 'n/a')}; "
            f"grader={grader}; "
            f"hallucinated_citations={count_hallucinated_citations(state)}; "
            f"errors={len(state.errors)}"
            + (f"; judge_note={verdict.weaknesses[:80]}" if verdict and verdict.weaknesses else "")
        ),
    )
    logger.info(
        "benchmark.done run=%s latency=%.2fs quality=%.1f coverage=%.0f%%",
        run_name,
        latency,
        metrics.quality_score or 0.0,
        (metrics.citation_coverage or 0.0) * 100,
    )
    return state, metrics
