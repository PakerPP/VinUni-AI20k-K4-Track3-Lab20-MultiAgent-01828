"""Benchmark report rendering."""

from __future__ import annotations

from datetime import UTC, datetime

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to markdown."""

    lines = [
        "# Benchmark Report",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        citation = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} "
            f"| {citation} | {failure} | {item.notes} |"
        )
    return "\n".join(lines) + "\n"


def render_full_report(
    metrics: list[BenchmarkMetrics],
    states: dict[str, ResearchState] | None = None,
    query_list: list[str] | None = None,
) -> str:
    """Render the deliverable report: comparison table, trace, and analysis."""

    states = states or {}
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    parts = [
        "# Benchmark Report: Single-Agent vs Multi-Agent",
        "",
        f"_Generated: {generated}_",
        "",
        "## 1. Setup",
        "",
        "- **Evidence source:** offline corpus `ai_agent_offline_research_corpus_v2/` "
        "(30 topics, web search disabled per the corpus benchmark rule).",
        "- **Baseline (control):** one LLM call over the **same evidence** as the crew "
        "- so the comparison measures orchestration, not retrieval.",
        "- **Baseline (no retrieval):** contrast arm with no evidence at all; its 0% "
        "citation coverage is an artefact of having nothing to cite.",
        "- **Multi-agent:** Supervisor routes Researcher → Analyst → Writer → Critic.",
        "- **Citation rule:** answers may cite only `source_id` / `article_id` values "
        "present in the retrieved evidence.",
    ]
    if query_list:
        parts += ["", "**Queries benchmarked:**", ""]
        parts += [f"{i}. {q}" for i, q in enumerate(query_list, 1)]

    # Strip the standalone "# Benchmark Report" heading so it does not nest here.
    table_lines = [
        line
        for line in render_markdown_report(metrics).strip().splitlines()
        if not line.startswith("# ")
    ]
    table_only = "\n".join(table_lines).strip()
    parts += ["", "## 2. Results", "", table_only, ""]

    # Aggregate comparison between the two modes.
    # "baseline-noretrieval-*" must not fall into the baseline bucket - it would
    # drag the control arm's citation coverage toward zero and flatter the crew.
    naive = [m for m in metrics if m.run_name.startswith("baseline-noretrieval")]
    base = [
        m
        for m in metrics
        if m.run_name.startswith("baseline") and not m.run_name.startswith("baseline-noretrieval")
    ]
    multi = [m for m in metrics if m.run_name.startswith("multi")]
    if base and multi:

        def avg(items: list[BenchmarkMetrics], attr: str) -> float:
            vals = [getattr(i, attr) or 0.0 for i in items]
            return sum(vals) / len(vals) if vals else 0.0

        header = "| Metric | Single-agent | Multi-agent | Delta |"
        divider = "|---|---:|---:|---:|"
        if naive:
            header = "| Metric | Single-agent | Multi-agent | Delta | No-retrieval |"
            divider = "|---|---:|---:|---:|---:|"

        parts += ["## 3. Aggregate comparison", "", header, divider]
        for label, attr, fmt in [
            ("Avg latency (s)", "latency_seconds", "{:.2f}"),
            ("Avg cost (USD)", "estimated_cost_usd", "{:.4f}"),
            ("Avg quality (0-10)", "quality_score", "{:.1f}"),
            ("Avg citation coverage", "citation_coverage", "{:.0%}"),
        ]:
            b, m = avg(base, attr), avg(multi, attr)
            delta = m - b
            sign = "+" if delta >= 0 else ""
            row = f"| {label} | {fmt.format(b)} | {fmt.format(m)} | {sign}{fmt.format(delta)} |"
            if naive:
                row += f" {fmt.format(avg(naive, attr))} |"
            parts.append(row)
        parts.append("")

        if naive:
            parts += [
                "> The **no-retrieval** column is the arm an earlier version of this benchmark",
                "> used as its control. Comparing the crew against it produces a large citation",
                "> gap that measures retrieval rather than orchestration. It is reported here",
                "> only to show why that comparison is misleading.",
                "",
            ]

    # Route traces make the "explain who did what" rubric item answerable.
    if states:
        parts += ["## 4. Trace summary", ""]
        for name, state in states.items():
            parts += [
                f"### {name}",
                "",
                f"- Routes: `{' → '.join(state.route_history) or 'n/a'}`",
                f"- Iterations: {state.iteration}",
                f"- Sources retrieved: {len(state.sources)}",
                f"- Errors: {state.errors or 'none'}",
            ]
            cites = [
                str(d.metadata.get("source_id"))
                for d in state.sources
                if d.metadata.get("source_id")
            ]
            if cites:
                parts.append(f"- Evidence ids: {', '.join(cites)}")
            synth = [
                str(d.metadata.get("source_id"))
                for d in state.sources
                if d.metadata.get("is_synthetic")
            ]
            if synth:
                parts.append(f"- Synthetic (fictional) evidence present: {', '.join(synth)}")
            parts.append("")

    return "\n".join(parts) + "\n"
