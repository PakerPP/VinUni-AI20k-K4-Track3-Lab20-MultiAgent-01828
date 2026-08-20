"""Command-line entrypoint for the multi-agent research lab."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.baseline import SingleAgentBaseline
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.judge import AnswerJudge
from multi_agent_research_lab.evaluation.report import render_full_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import (
    configure_tracing,
    export_trace_json,
    get_provider,
    render_trace_table,
    reset_spans,
)
from multi_agent_research_lab.services.llm_client import LLMClient, MockLLMClient
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()

DEFAULT_CONFIG = Path("configs/lab_default.yaml")


def _init() -> None:
    configure_logging(get_settings().log_level)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


def _make_llm(mock: bool) -> object:
    """Mock client when requested or when no API key is configured."""

    settings = get_settings()
    if mock or not settings.openai_api_key:
        if not mock:
            console.print("[yellow]No OPENAI_API_KEY found - using mock LLM.[/yellow]")
        return MockLLMClient()
    return LLMClient()


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    mock: Annotated[bool, typer.Option("--mock", help="Use the offline mock LLM")] = False,
    max_sources: Annotated[int, typer.Option("--max-sources", help="Evidence to retrieve")] = 5,
    retrieval: Annotated[
        bool,
        typer.Option(
            "--retrieval/--no-retrieval",
            help="Give the baseline the same evidence as the crew (default) or none",
        ),
    ] = True,
) -> None:
    """Run the single-agent baseline: one LLM call over the same evidence as the crew."""

    _init()
    request = _parse_query(query)
    state = SingleAgentBaseline(llm_client=_make_llm(mock), retrieval=retrieval).run(
        request.query, max_sources=max_sources
    )
    if state.errors:
        console.print(Panel.fit("\n".join(state.errors), title="Errors", style="red"))
    console.print(
        Panel.fit(
            Text(state.final_answer or "(no answer)"),
            title="Single-Agent Baseline",
        )
    )


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    mock: Annotated[bool, typer.Option("--mock", help="Use the offline mock LLM")] = False,
    max_sources: Annotated[int, typer.Option("--max-sources", help="Evidence to retrieve")] = 5,
    trace_out: Annotated[
        Path | None, typer.Option("--trace-out", help="Write a JSON trace here")
    ] = None,
    show_trace: Annotated[
        bool,
        typer.Option("--show-trace/--no-show-trace", help="Print the trace table"),
    ] = True,
    tracer: Annotated[
        str, typer.Option("--tracer", help="Tracing provider: auto|otel|langsmith|none")
    ] = "auto",
    screenshot: Annotated[
        Path | None, typer.Option("--screenshot", help="Render the trace to a PNG here")
    ] = None,
) -> None:
    """Run the multi-agent workflow: Supervisor, Researcher, Analyst, Writer, Critic."""

    _init()
    reset_spans()
    if tracer != "none":
        provider = configure_tracing(tracer)
        console.print(f"[dim]tracing provider: {provider.name}[/dim]")
    request = ResearchQuery(query=_parse_query(query).query, max_sources=max_sources)
    state = ResearchState(request=request)

    workflow = MultiAgentWorkflow(llm_client=_make_llm(mock))
    state = workflow.run(state)

    console.print(Panel.fit(Text(state.final_answer or "(no answer)"), title="Multi-Agent Answer"))

    critic = [r for r in state.agent_results if str(r.agent) == "critic"]
    if critic:
        console.print(Panel.fit(Text(critic[0].content), title="Critic Review", style="cyan"))

    if state.errors:
        console.print(Panel.fit("\n".join(state.errors), title="Errors", style="red"))

    if show_trace:
        console.print(Panel.fit(render_trace_table(state), title="Trace", style="dim"))

    if trace_out:
        path = export_trace_json(state, trace_out, run_name="multi-agent")
        console.print(f"[green]Trace written to {path}[/green]")

    if screenshot:
        from multi_agent_research_lab.observability.screenshot import render_trace_png

        try:
            png = render_trace_png(state, screenshot)
            console.print(f"[green]Trace screenshot written to {png}[/green]")
        except Exception as exc:  # a missing plot backend must not fail the run
            console.print(f"[yellow]Could not render screenshot: {exc}[/yellow]")

    active = get_provider()
    if hasattr(active, "finished_spans"):
        console.print(f"[dim]{active.name}: exported {len(active.finished_spans())} spans[/dim]")


@app.command()
def benchmark(
    config: Annotated[Path, typer.Option("--config", help="YAML config")] = DEFAULT_CONFIG,
    mock: Annotated[bool, typer.Option("--mock", help="Use the offline mock LLM")] = False,
    out: Annotated[Path, typer.Option("--out", help="Report path")] = Path(
        "reports/benchmark_report.md"
    ),
    limit: Annotated[int, typer.Option("--limit", help="Max queries to run")] = 0,
    judge: Annotated[
        bool, typer.Option("--judge/--no-judge", help="Grade answers with an LLM judge")
    ] = True,
    naive: Annotated[
        bool,
        typer.Option(
            "--naive/--no-naive",
            help="Also run a no-retrieval baseline to show why that comparison misleads",
        ),
    ] = True,
) -> None:
    """Benchmark single-agent vs multi-agent across the configured queries."""

    _init()
    if not config.is_file():
        console.print(f"[red]Config not found: {config}[/red]")
        raise typer.Exit(code=1)

    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    queries: list[str] = cfg.get("benchmark", {}).get("queries", [])
    if limit > 0:
        queries = queries[:limit]
    if not queries:
        console.print("[red]No benchmark queries in config.[/red]")
        raise typer.Exit(code=1)

    all_metrics = []
    states: dict[str, ResearchState] = {}

    for i, query in enumerate(queries, 1):
        console.print(f"\n[bold]({i}/{len(queries)})[/bold] {query}")

        def _judge() -> AnswerJudge | None:
            # A separate client so judging cost never lands on the arm being measured.
            return AnswerJudge(_make_llm(mock)) if judge else None

        def _report(label: str, m: BenchmarkMetrics) -> None:
            console.print(
                f"  {label:<18}: {m.latency_seconds:5.2f}s "
                f"quality={m.quality_score:.1f} "
                f"coverage={(m.citation_coverage or 0):.0%}"
            )

        # Arm 1: baseline WITH the same evidence as the crew (the fair control).
        reset_spans()
        base_llm = _make_llm(mock)
        base = SingleAgentBaseline(llm_client=base_llm, retrieval=True)
        state_b, metrics_b = run_benchmark(
            f"baseline-{i}", query, base.run, base_llm, judge=_judge()
        )
        all_metrics.append(metrics_b)
        states[f"baseline-{i}"] = state_b
        _report("baseline", metrics_b)

        # Arm 2: multi-agent crew.
        reset_spans()
        multi_llm = _make_llm(mock)
        workflow = MultiAgentWorkflow(llm_client=multi_llm)

        def _run_multi(q: str, _wf: MultiAgentWorkflow = workflow) -> ResearchState:
            return _wf.run(ResearchState(request=ResearchQuery(query=q)))

        state_m, metrics_m = run_benchmark(
            f"multi-agent-{i}", query, _run_multi, multi_llm, judge=_judge()
        )
        all_metrics.append(metrics_m)
        states[f"multi-agent-{i}"] = state_m
        _report("multi-agent", metrics_m)

        # Arm 3 (optional): no-retrieval baseline, kept to show the misleading comparison.
        if naive:
            reset_spans()
            naive_llm = _make_llm(mock)
            naive_run = SingleAgentBaseline(llm_client=naive_llm, retrieval=False)
            state_n, metrics_n = run_benchmark(
                f"baseline-noretrieval-{i}", query, naive_run.run, naive_llm, judge=_judge()
            )
            all_metrics.append(metrics_n)
            states[f"baseline-noretrieval-{i}"] = state_n
            _report("baseline (no ret.)", metrics_n)

    report = render_full_report(all_metrics, states, queries)
    store = LocalArtifactStore(root=out.parent if out.parent.name else Path("reports"))
    path = store.write_text(out.name, report)

    table = Table(title="Benchmark Summary")
    for col in ("Run", "Latency (s)", "Cost (USD)", "Quality", "Citation cov."):
        table.add_column(col)
    for m in all_metrics:
        table.add_row(
            m.run_name,
            f"{m.latency_seconds:.2f}",
            "-" if m.estimated_cost_usd is None else f"{m.estimated_cost_usd:.4f}",
            f"{m.quality_score:.1f}" if m.quality_score is not None else "-",
            f"{(m.citation_coverage or 0):.0%}",
        )
    console.print(table)
    console.print(f"[green]Report written to {path}[/green]")


if __name__ == "__main__":
    app()
